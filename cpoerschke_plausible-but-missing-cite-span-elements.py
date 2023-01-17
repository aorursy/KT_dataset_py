import glob

directory_name = "/Users/cpoerschke/Downloads/CORD-19-research-challenge"

file_names = glob.glob("{}/*/*/*.json".format(directory_name))
print("Found {} files in the {} directory.".format(len(file_names), directory_name))
# simple illustrative id-based patterns for citations on their own (e.g. ' [1] ')
# or in a citation group (e.g. ' [2, ' and ' 4, ' and ' 8] ' within ' [2, 4, 8] ')
patterns = [
    [ " ", "", 0, "", " " ], # e.g. BIBREF9 cited as ' 9 '
    [ " ", "", 1, "", " " ], # e.g. BIBREF0 cited as ' 1 '

    [ " ", "[", 0, "]", " " ], # e.g. BIBREF9 cited as ' [9] '
    [ " ", "[", 1, "]", " " ], # e.g. BIBREF0 cited as ' [1] '

    [ " ", "", 0, ",", " " ], # e.g. BIBREF9 cited as ' 9, '
    [ " ", "", 1, ",", " " ], # e.g. BIBREF0 cited as ' 1, '

    [ " ", "[", 0, ",", " " ], # e.g. BIBREF9 cited as ' [9, '
    [ " ", "[", 1, ",", " " ], # e.g. BIBREF0 cited as ' [1, '

    [ " ", "", 0, "]", " " ], # e.g. BIBREF9 cited as ' 9] '
    [ " ", "", 1, "]", " " ], # e.g. BIBREF0 cited as ' 1] '
]

# given a pattern return the citation texts for an id
def bibref_texts(n, patterns_index):
    text_prefix2 = patterns[patterns_index][0]
    text_prefix1 = patterns[patterns_index][1]
    n_offset     = patterns[patterns_index][2]
    text_suffix1 = patterns[patterns_index][3]
    text_suffix2 = patterns[patterns_index][4]

    text1 =     "{}{}{}".format(              text_prefix1, n+n_offset, text_suffix1)
    text2 = "{}{}{}{}{}".format(text_prefix2, text_prefix1, n+n_offset, text_suffix1, text_suffix2)

    # example input: n=42
    # example output: text1='[42]' text2=' [42] '
    return text1, text2

# given a citation in a text, which pattern(s) does it match?
def get_patterns_indices(ref_id, text, start, end):
    indices = set()
    for index in range(0,len(patterns)):
        text1, text2 = bibref_texts(ref_id, index)
        if text1 == text[start:end] and text2 == text[start-1:end+1]:
            indices.add(index)
    return indices

# convert e.g. 'BIBREF42' string into '42' number
def ref_id_from_bib_entry_key(key):
    return int(key[len("BIBREF"):])

# convert e.g. '42' number back into into 'BIBREF42' string
def bib_entry_key_from_ref_id(ref_id):
    return "BIBREF{}".format(ref_id)

# identify bib entries not referenced in any of the document's sections' cite spans
def extract_unreferenced_ids(obj):
    
    # ids listed in the bib_entries
    reference_ids = set([ ref_id_from_bib_entry_key(bib_entry_key) for bib_entry_key in obj["bib_entries"].keys() ])

    # ids referenced in the text sections
    referenced_ids = set()
    for section_key in ["abstract", "body_text", "back_matter"]:
        for section in obj[section_key]:
            for span in section["cite_spans"]:
                if span["ref_id"] is not None:
                    referenced_ids.add(ref_id_from_bib_entry_key(span["ref_id"]))
                    
    # ids not referenced in the text sections
    return reference_ids.difference(referenced_ids)

# identify ...
def propose_cite_spans(obj, unreferenced_ids, verbose=False):

    proposed_cite_spans = []
    
    if verbose:
        print("unreferenced_ids={}".format(unreferenced_ids))

    # find recognised citation style patterns used in the document
    used_citation_styles = set()
    
    for section_key in ["abstract", "body_text", "back_matter"]:
        for section in obj[section_key]:
            for span in section["cite_spans"]:
                if span["ref_id"] is not None:
                    # simple illustrative id-based pattern matching
                    patterns_indices = get_patterns_indices(\
                        ref_id_from_bib_entry_key(span["ref_id"]), \
                        section["text"], span["start"], span["end"])

                    # if the existing cite span is not in a recognised style
                    if len(patterns_indices) == 0:
                        if verbose:
                            print("unrecognised citation style: span={}".format(span))
                        # then return without proposing additional cite spans
                        return proposed_cite_spans
                    
                    # accumulate the citation styles used in the document
                    used_citation_styles.update(patterns_indices)

    if verbose:
        print("used_citation_styles={}".format(used_citation_styles))
        
    # which not-yet-referenced ids can be matched to the document's text?
    # (via a citation style pattern style pattern already used in the document)
    unreferenced_id_to_citation_styles = {}
    
    for ref_id in unreferenced_ids:
        unreferenced_id_to_citation_styles[ref_id] = []
        for patterns_index in used_citation_styles:
            # example input: ref_id=42
            text1, text2 = bibref_texts(ref_id, patterns_index)
            # example output: text1='[42]' text2=' [42] '
            for section_key in ["abstract", "body_text", "back_matter"]:
                for section in obj[section_key]:
                    # example: find the first ' [42] ' occurrence
                    pos = section["text"].find(text2)
                    while 0 <= pos:
                        # record the find
                        unreferenced_id_to_citation_styles[ref_id].append(patterns_index)
                        # find any additional occurrences
                        pos = section["text"].find(text2, pos + len(text2))
            
    if verbose:
        print("unreferenced_id_to_citation_styles={}".format(unreferenced_id_to_citation_styles))

    # which not-yet-referenced ids can be unambiguously matched to the document's text?
    unreferenced_id_to_citation_style = {}
    for (ref_id, citation_styles) in unreferenced_id_to_citation_styles.items():
        if len(citation_styles) == 1:
            unreferenced_id_to_citation_style[ref_id] = citation_styles[0]
            
    if verbose:
        print("unreferenced_id_to_citation_style={}".format(unreferenced_id_to_citation_style))
    
    # now combine unreferenced ids + citation styles + text + bib_entries into proposed cite spans
    for (ref_id, patterns_index) in unreferenced_id_to_citation_style.items():
        bib_entries_key = bib_entry_key_from_ref_id(ref_id)

        for section_key in ["abstract", "body_text", "back_matter"]:
            for section in obj[section_key]:                
                
                text1, text2 = bibref_texts(ref_id, patterns_index)
                pos1, pos2 = section["text"].find(text1), section["text"].find(text2)
                # given the location of the proposed citation
                if 0 <= pos1 and 0 <= pos2:                                    
                    # obtain some surrounding text
                    prev_dot_pos = section["text"].rfind(".", 0, pos2-1)
                    next_dot_pos = section["text"].find(".", pos2+len(text2))
                    surrounding_text = section["text"][prev_dot_pos+1:next_dot_pos+1].lstrip().rstrip()
                    # fill in a proposed cite span element
                    proposed_cite_span = {
                        "start" : pos1,
                        "end" : pos1 + len(text1),
                        "text" : text1,
                        "explanation" : { # extra element, not present in "cite_span" schema
                            "surrounding_text" : surrounding_text,
                            "bib_entry" : obj["bib_entries"][bib_entries_key],
                        },
                        "ref_id" : bib_entries_key,
                    }
                    # add the proposed cite span element to the section
                    if "proposed_cite_spans" not in section:
                        section["proposed_cite_spans"] = []
                    section["proposed_cite_spans"].append(proposed_cite_span)
                    
                    # for convenience also directly return the proposed element
                    proposed_cite_spans.append(proposed_cite_span)
    
    return proposed_cite_spans
import json

verbose_count = 0
num_proposed_total = 0
proposed_obj_count = 0

unreferenced_ids_limit = 3

print("Considering documents with up to {} unreferenced bib entries.".format(unreferenced_ids_limit))

for file_name in file_names:
    with open(file_name, 'r') as file:
        obj = json.loads(file.read())         
        unreferenced_ids = extract_unreferenced_ids(obj)
        
        if len(unreferenced_ids) > unreferenced_ids_limit:
            continue
            
        paper_id = obj["paper_id"]
        title = obj["metadata"]["title"]
            
        verbose = paper_id in [
            "b7b03c0cfccc9f607bef14d153097bf57ed2ca67",
            "6f8725026d83e6829275eefc91aef6fbdacc9c06",
            "f63ada8bc0784ce879c45f3d68ce08b67d4a8b6a",
            "81c77f6ea4ad20908978c014fea0abb7d5c557aa"
        ]
            
        if verbose:
            verbose_count += 1
            print()
            print("\"paper_id\": {}".format(paper_id))
            print("\"title\": {}".format(title))
     
        proposed_cite_spans = propose_cite_spans(obj, unreferenced_ids, verbose=verbose)
        
        if verbose:
            print("\"proposed_cite_spans\": {}".format(json.dumps(proposed_cite_spans, indent=2)))
            print()
        
        if len(proposed_cite_spans) != 0:
            num_proposed_total += len(proposed_cite_spans)
            proposed_obj_count += 1

print("Proposed {} cite spans in {} documents. "\
      "Verbose output was shown for {} documents.".format(num_proposed_total, proposed_obj_count, verbose_count))