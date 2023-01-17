!pip install -q pandas==1.0.3 spacy==2.2.1 scispacy==0.2.4 https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz numpy==1.18.1
import json

import warnings

import shutil

import en_core_sci_md

import pandas as pd

from enum import Enum

from spacy.tokens import Doc, Token, Span

from pathlib import Path

from typing import Tuple, Dict, Any, Optional, List, Set

from pprint import pprint



CORD19_INPUT_DIR = Path("/kaggle/input/CORD-19-research-challenge")

ANNOTATIONS_INPUT_FILE = Path("/kaggle/input/generate-umls-annotations/umls_annotations.zip")

ANNOTATION_DIR = Path("/tmp/umls_annotations")

ANNOTATION_DIR.mkdir(exist_ok=True, parents=True)

shutil.unpack_archive(ANNOTATIONS_INPUT_FILE, ANNOTATION_DIR.parent)



OUTPUT_DIR = Path("/kaggle/working")
for annotation_file in ANNOTATION_DIR.iterdir():

    with open(annotation_file, "r") as f:

        paper_annotations = json.load(f)

        

    if len(paper_annotations) > 0:

        pprint((annotation_file.name, paper_annotations[:2]))

        break
def all_json_iter() -> Tuple[str, Dict[str, Any]]:

    """

    Iterate over all data files across all text subsets

    """

    all_files = CORD19_INPUT_DIR.glob(

        "document_parses/*/*.json"

    )



    for json_file in all_files:

        if json_file.name.startswith("."):

            # There are some garbage files in here for some reason

            continue



        with open(json_file, "r", encoding="utf-8") as f:

            try:

                article_json = json.load(f)

            except Exception:

                raise RuntimeError(f"Failed to parse json from {json_file}")



        # PMC XML, PDF, etc

        text_type = json_file.parent.name



        yield text_type, article_json

        

        

def get_annotations(sha: str) -> List[Dict[str, Any]]:

    with open(ANNOTATION_DIR / f"{sha}.json", "r") as f:

        return json.load(f)

        

        

def get_article_json(sha: str) -> Dict[str, Any]:

    article_files = list(CORD19_INPUT_DIR.glob(f"document_parses/*/{sha}.json"))

    if len(article_files) == 0:

        raise RuntimeError(f"No JSON file found for SHA {sha}")

    else:

        # If there are multiple parses for this document, we'll take the last one

        # This is intended to match up with how the annotations are generated --

        # If there are multiple papers with the same ID (SHA), the last one will end

        # up in the annotations

        article_file = article_files[-1]

        

    with open(article_file, "r") as f:

        return json.load(f)

        

def get_paper_id(article_json: Dict[str, Any]) -> str:

    return article_json["paper_id"]





def get_title(article_json: Dict[str, Any]) -> str:

    return article_json["metadata"]["title"]





def get_abstract(article_json: Dict[str, Any]) -> str:

    if "abstract" not in article_json:

        return ""

    return "\n\n".join(a["text"] for a in article_json["abstract"])





def get_full_text(article_json: Dict[str, Any]) -> str:

    if "body_text" not in article_json:

        return ""

    return "\n\n".join(a["text"] for a in article_json["body_text"])





def get_all_text(article_json: Dict[str, Any]) -> str:

    return f"{get_abstract(article_json)} {get_full_text(article_json)}"
filtered_annotations = {}



COVID19_TERMS = tuple(t.lower() for t in (

    "covid",

    "covid-",

    "SARS-CoV-2",

    "HCoV-19",

    "coronavirus 2",

))



HYPERCOAG_CUIS = set((

    "C0398623",  # Thrombophilia

    "C2984172",  # F5 Leiden Allele

    "C0311370",  # Lupus anticoagulant disorder

    "C1704321",  # Nephrotic Syndrome, Minimal Change

    "C3202971",  # Non-Infective Endocarditis

    "C0040053",  # Thrombosis

    "C2826333",  # D-Dimer Measurement

    "C0060323",  # Fibrin fragment D

    "C3536711",  # Anti-coagulant [EPC]

    "C2346807",  # Anti-Coagulation Factor Unit

    "C0012582",  # dipyridamole

))



num_papers_missing_text = 0

num_annotations_found = 0

num_annotations_missing = 0



metadata = pd.read_csv(CORD19_INPUT_DIR / "metadata.csv")

shas = set()

publish_times = {}

dois = {}

# The sha field may have multiple SHAs delimited by semicolons

for multi_sha, publish_time, doi in zip(metadata["sha"], metadata["publish_time"], metadata["doi"]):

    if not pd.isnull(multi_sha):

        for sha in multi_sha.split("; "):

            shas.add(sha)

            publish_times[sha] = publish_time

            dois[sha] = doi



for sha in shas:

    try:

        paper_json = get_article_json(sha)

    except RuntimeError:

        # This paper doesn't have a JSON file for its full text

        num_papers_missing_text += 1

        continue

        

    paper_annotations = get_annotations(sha)

    if paper_annotations is None:

        # Missing annotations here indicates a desync between our generated annotations

        # and the input data -- keep track but continue

        num_annotations_missing += 1

        continue

    else:

        num_annotations_found += 1

    

    found_covid19 = False

    found_hypercoag = False

    

    search_text = f"{get_title(paper_json)}\n{get_abstract(paper_json)}".lower()

    

    for term in COVID19_TERMS:

        if term in search_text:

            found_covid19 = True

            break

    if not found_covid19:

        continue

    

    paper_concepts = set(concept["cui"] for concept in paper_annotations)

    for hypercoag_cui in HYPERCOAG_CUIS:

        if hypercoag_cui in paper_concepts:

            found_hypercoag = True

            break

            

    if found_covid19 and found_hypercoag:

        filtered_annotations[sha] = paper_annotations[:]





print(f"Checked {num_annotations_found} paper parses with annotations.\n"

      f"Ignored {num_annotations_missing} paper parses without annotations and {num_papers_missing_text} papers without text available.\n"

      f"Identified {len(filtered_annotations)} papers related to COVID-19 and hypercoagulability.")
with open(OUTPUT_DIR / "filtered_annotations.json", "w") as f:

    json.dump(filtered_annotations, f)
nlp = en_core_sci_md.load()
# https://stackoverflow.com/a/493788

def text2int(textnum, numwords={}):

    """

    Convert anything that matches the spaCy "like_num" rule to an integer.

    

    https://spacy.io/api/token#attributes

    """

    try:

        # Commas trip up the int parser, so remove them if there are any

        return int(textnum.replace(",", ""))

    except ValueError:

        pass

    

    if not numwords:

        units = [

          "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",

          "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",

          "sixteen", "seventeen", "eighteen", "nineteen",

        ]



        tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]



        scales = ["hundred", "thousand", "million", "billion", "trillion"]



        numwords["and"] = (1, 0)

        for idx, word in enumerate(units):

            numwords[word] = (1, idx)

        for idx, word in enumerate(tens):

            numwords[word] = (1, idx * 10)

        for idx, word in enumerate(scales):

            numwords[word] = (10 ** (idx * 3 or 2), 0)



    current = result = 0

    for word in textnum.split():

        if word not in numwords:

            raise ValueError("Illegal word: " + word)



        scale, increment = numwords[word]

        current = current * scale + increment

        if scale > 100:

            result += current

            current = 0



    return result + current



def get_span(annotation: Dict[str, Any], doc: Doc) -> Span:

    """

    Return the span corresponding to the given annotation.

        

    Assumes the start/end indices in the annotation line up correctly with the document

    (i.e., the document was constructed exactly the same way in the original annotation process

    as it was in the given parsed document).

    """

    return doc.char_span(annotation["start"], annotation["end"])



def get_context(annotation: Dict[str, Any], doc: Doc) -> str:

    """

    Return the context (sentence) in the document containing the given annotation.

    """

    return get_span(annotation, doc).sent.text



def get_root_token(annotation: Dict[str, Any], doc: Doc) -> Token:

    """

    Return the root token for the given annotation.

    """

    return get_span(annotation, doc).root
SAMPLE_SIZE_NOUNS = set(("patient", "subject", "case", "birth"))



def find_sample_size(doc: Doc) -> Tuple[Optional[int], Optional[str], Optional[str]]:

    """

    For a parsed spaCy document representing a paper, try to identify the sample size heuristically.

    If possible, return a tuple containing the sample size, noun describing

    the sample, and the sentence that inference was generated from.

    

    If not, return None for each tuple element.

    """

    sample_size_candidates = []

    

    for tok in doc:

        if tok.like_num and tok.head.lemma_ in SAMPLE_SIZE_NOUNS and tok.dep_ == "nummod":

            try:

                sample_size_int = text2int(tok.text.lower())

            except ValueError:

                continue

            sample_size_candidates.append((sample_size_int, tok.head.text, tok.sent.text))

    

    if len(sample_size_candidates) == 0:

        return (None, None, None)

    chosen_candidate = max(sample_size_candidates, key=lambda c: c[0])

    return chosen_candidate

SYS_REV_CUIS = set((

    "C1955832",  # Review, Systematic

    "C0282458",  # Meta-Analysis (publications)

))



EXP_REV_CUIS = set((

    "C0282443",  # Review [Publication Type]

))



SIM_CUIS = set((

    "C0376284",  # Machine Learning

    "C0683579",  # scenario

))



RET_OBS_CUIS = set((

    "C0035363",  # Retrospective Studies

    "C2362543",  # Electronic Health Records

    "C2985505",  # Retrospective Cohort Study

))



OBS_CUIS = set((

    "C0030705",  # Patients

))



EDIT_WORDS = set((

    "editor",

    "opinion",

))



RET_OBS_WORDS = set((

    "retrospective",

    "retrospectively",

    "autopsy",

))



PROS_OBS_WORDS = set((

    "prospective",

    "prospectively",

    "enrolled",

))





class StudyType(Enum):

    PROS_OBS = "Prospective observational study"

    RET_OBS = "Retrospective observational study"

    OBS = "Observational study"

    SYS_REV = "Systematic review and meta-analysis"

    EXP_REV = "Expert review"

    SIM = "Simulation"

    EDIT = "Editorial"



def find_study_type(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str], Optional[str]]:

    """

    Attempt to identify study type using words and concepts in the document.

    """

    study_type = None

    study_type_words = set()

    contexts = []

    # Fallback classification for studies which are clearly observational, but we can't

    # tell whether it's prospective or retrospective

    is_observational = False



    doc_annotations = {a["cui"]: a for a in annotations}

    

    def update_study_type(new_study_type: StudyType, word: str, context: str):

        nonlocal study_type

        study_type = new_study_type

        study_type_words.add(word)

        contexts.append(context)

        

    def check_cui_set(cui_set: Set[str], cui_set_study_type: StudyType):

        for cui in cui_set:

            if cui in doc_annotations:

                annotation = doc_annotations[cui]

                update_study_type(cui_set_study_type,

                                  annotation["canonical_name"],

                                  get_context(annotation, doc))

                

    def check_word_set(word_set: Set[str], word_set_study_type: StudyType):

        for sent in doc.sents:

            for tok in sent:

                if tok.lemma_.lower() in word_set:

                    update_study_type(word_set_study_type, tok.text, sent.text)

        

    # Check CUIs first, since those should be more reliable

    for cui_set, cui_set_study_type in (

        (SYS_REV_CUIS, StudyType.SYS_REV),

        (EXP_REV_CUIS, StudyType.EXP_REV),

        (SIM_CUIS, StudyType.SIM),

        (RET_OBS_CUIS, StudyType.RET_OBS),

    ):

        if study_type is None:

            check_cui_set(cui_set, cui_set_study_type)

    

    # Check word sets next

    for word_set, word_set_study_type in (

        (EDIT_WORDS, StudyType.EDIT),

        (RET_OBS_WORDS, StudyType.RET_OBS),

        (PROS_OBS_WORDS, StudyType.PROS_OBS)

    ):

        if study_type is None:

            check_word_set(word_set, word_set_study_type)

            

    # Finally, if we still don't have a study type, check the fallback umbrella study type

    # for all observational studies

    if study_type is None:

        check_cui_set(OBS_CUIS, StudyType.OBS)

        

    return (

        None if study_type is None else study_type.value,

        "; ".join(list(study_type_words)),

        "\n".join(contexts)

    )
SEVERITY_WORDS = set(("mild", "severe", "critical", "ICU", "intensive care unit"))



def find_severity(doc: Doc) -> Tuple[Optional[str], Optional[str]]:

    """

    Attempt to find case severity in the document.

    """

    severity = set()

    contexts = []

    

    for tok in doc:

        if tok.lemma_ in SEVERITY_WORDS:

            severity.add(tok.text.lower())

            contexts.append(tok.sent.text)

            

    if len(severity) == 0:

        return None, None



    return "; ".join(list(severity)), "\n".join(contexts)
THERAPEUTIC_METHOD_CUIS = set((

    "C0012582",  # dipyridamole

    "C1963724",  # Antiretroviral therapy

    "C0770546",  # heparin, porcine

))



def find_therapeutic_methods(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:

    """

    Attempt to find therapeutic methods by UMLS concepts in the document.

    """

    methods = set()

    contexts = []

    

    for annotation in annotations:

        if annotation["cui"] in THERAPEUTIC_METHOD_CUIS:

            methods.add(annotation["canonical_name"])

            contexts.append(get_context(annotation, doc))

            

    return (

        "; ".join(list(methods)),

        "\n".join(contexts),

    )
OUTCOME_CUIS = set((

    "C0332281",  # Associated with

    "C0392756",  # Reduced

    "C1260953",  # Suppressed

    "C0309872",  # PREVENT (product) [proxy for word "prevent"]

    "C0278252",  # Prognosis bad

    "C0035648",  # risk factors

    "C0184511",  # Improved

    "C0442805",  # Increased

    "C0205216",  # Decreased

))



def find_outcomes(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:

    """

    Attempt to find outcome excerpts by UMLS concepts in the document.

    """

    outcome_words = set()

    contexts = []

    

    for annotation in annotations:

        if annotation["cui"] in OUTCOME_CUIS:

            outcome_words.add(annotation["canonical_name"])

            contexts.append(get_context(annotation, doc))

            

    return (

        "; ".join(list(outcome_words)),

        "\n".join(contexts),

    )

ENDPOINT_CUIS = set((

    "C0011065",  # Cessation of life

    "C0026565",  # Mortality Vital Statistics

))



def find_endpoints(doc: Doc, annotations: List[Dict[str, Any]]) -> Tuple[Optional[str], Optional[str]]:

    """

    Attempt to find information related to primary endpoints by UMLS concepts in the document.

    """

    endpoint_words = set()

    contexts = []

    

    for annotation in annotations:

        if annotation["cui"] in ENDPOINT_CUIS:

            endpoint_words.add(annotation["canonical_name"])

            contexts.append(get_context(annotation, doc))

            

    return (

        "; ".join(list(endpoint_words)),

        "\n".join(contexts),

    )
GOOD_MECHANISM_CUIS = set((

    "C2247948",  # response to type I interferon

    "C0005821",  # Blood Platelets

    "C0200635",  # Lymphocyte Count measurement

    "C0301863",  # "U" lymphocyte

    "C1556326",  # Adverse Event Associated with Coagulation

    "C0019010",  # Hemodynamics

    "C1527144",  # Therapeutic Effect

    

))



BAD_MECHANISM_CUIS = set((

    "C1883725",  # Replicate

    "C0042774",  # Virus Replication

    "C0677042",  # Pathology processes

    "C0398623",  # Thrombophilia

    "C2826333",  # D-Dimer Measurement

    "C1861172",  # Venous Thromboembolism

))



UP_WORDS = set((

    "elicit",

    "good",

))



UP_CUIS = set((

    "C0442805",  # Increase

    "C0205250",  # High

    "C2986411",  # Improvement

))



DOWN_WORDS = set((

    "ameliorate",

))



DOWN_CUIS = set((

    "C1260953",  # Suppressed

    "C0205216",  # Decreased

    "C0392756",  # Reduced

    "C1550458",  # Abnormal

))



def find_clinical_improvement(

    doc: Doc,

    annotations: List[Dict[str, Any]]

) -> Tuple[Optional[str], Optional[str], Optional[str]]:

    """

    Identify phrases corresponding to clinical improvement (or not).  Return a tuple with 3 strings:

    "y/n" based on whether more evidence for improvement or worsening is found, evidence for improvement, and

    evidence for worsening.

    """

    improvement_evidence = set()

    worsening_evidence = set()

    

    # Locate tokens corresponding to all categories (good/bad mechanisms, up/down)

    bad_spans = set()

    good_spans = set()

    up_spans = set()

    down_spans = set()

    

    for tok in doc:

        for word_set, span_set in (

            (UP_WORDS, up_spans),

            (DOWN_WORDS, down_spans),

        ):

            if tok.lemma_ in word_set:

                span_set.add(doc[tok.i:tok.i+1])

    

    for annotation in annotations:

        for cui_set, span_set in (

            (GOOD_MECHANISM_CUIS, good_spans),

            (BAD_MECHANISM_CUIS, bad_spans),

            (UP_CUIS, up_spans),

            (DOWN_CUIS, down_spans),

        ):

            if annotation["cui"] in cui_set:

                annotation_span = get_span(annotation, doc)

                span_set.add(annotation_span)

    

    # Check dependencies to see if any of the discovered tokens relate in ways that

    # might provide evidence one for improvement or worsening

    for mechanism_set, modifier_set, evidence_set in (

        (bad_spans, up_spans, worsening_evidence),

        (bad_spans, down_spans, improvement_evidence),

        (good_spans, up_spans, improvement_evidence),

        (good_spans, down_spans, worsening_evidence),

    ):

        for mechanism_span in mechanism_set:

            mechanism_tok = mechanism_span.root

            for modifier_span in modifier_set:

                modifier_tok = modifier_span.root

                if modifier_tok in mechanism_tok.children:

                    evidence_set.add((modifier_span, mechanism_span))

            

    improvement = None

    if len(improvement_evidence) > len(worsening_evidence):

        improvement = "y"

    elif len(improvement_evidence) < len(worsening_evidence):

        improvement = "n"

        

    def format_evidence(evidence_set):

        return "; ".join(

            " ".join((modifier_span.text, mechanism_span.text)) for modifier_span, mechanism_span in evidence_set

        )

        

    return (

        improvement,

        format_evidence(improvement_evidence),

        format_evidence(worsening_evidence),

    )
auto_data = []



for sha, paper_annotations in filtered_annotations.items():

    try:

        article_json = get_article_json(sha)

    except RuntimeError:

        warnings.warn(f"No article JSON found for SHA {sha}")

        continue

    

    doc_text = f"{get_title(article_json)}\n\n{get_abstract(article_json)}"

    doc = nlp(doc_text)

    

    publish_time = publish_times[sha]

    doi = dois[sha]

    sample_size, sample_unit, sample_size_context = find_sample_size(doc)

    severity, severity_context = find_severity(doc)

    therapeutic_methods, therapeutic_method_context = find_therapeutic_methods(doc, paper_annotations)

    study_type, study_type_words, study_type_context = find_study_type(doc, paper_annotations)

    outcome_words, outcome_context = find_outcomes(doc, paper_annotations)

    endpoints, endpoint_context = find_endpoints(doc, paper_annotations)

    improvement, improvement_evidence, worsening_evidence = find_clinical_improvement(doc, paper_annotations)



    auto_data.append({

        "Paper ID": get_paper_id(article_json),

        "Title": get_title(article_json),

        "Abstract": get_abstract(article_json),

        "DOI": doi,

        "Date": publish_time,

        "Sample Size": sample_size,

        "Sample Unit": sample_unit,

        "Sample Size Context": sample_size_context,

        "Severity": severity,

        "Severity Context": severity_context,

        "Therapeutic Methods": therapeutic_methods,

        "Therapeutic Method Context": therapeutic_method_context,

        "Study Type": study_type,

        "Study Type Words": study_type_words,

        "Study Type Context": study_type_context,

        "Outcome Words": outcome_words,

        "Outcome Context": outcome_context,

        "Endpoint Words": endpoints,

        "Endpoint Context": endpoint_context,

        "Clinical Improvement": improvement,

        "Clinical Improvement Evidence": improvement_evidence,

        "Clinical Worsening Evidence": worsening_evidence,

    })

    

auto_df = pd.DataFrame(auto_data).set_index("Paper ID")

auto_df.to_csv(OUTPUT_DIR / "What is the efficacy of novel therapeutics being tested currently_.csv")