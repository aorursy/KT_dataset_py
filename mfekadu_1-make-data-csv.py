# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

from pprint import pprint as pp

from bs4 import BeautifulSoup, Tag
import re
from enum import Enum

# for garbage collection
import gc

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
!tar -xvzf /kaggle/input/stanford-plato-corpus/plato_mirror_spr2020.tgz
!ls stanford.library.sydney.edu.au
!ls -lah stanford.library.sydney.edu.au/entries | head -10
!ls -lah stanford.library.sydney.edu.au/entries/abduction
MAIN_FOLDER = "stanford.library.sydney.edu.au/entries/"
article_folders = [  os.path.join(MAIN_FOLDER, fol) for fol in os.listdir(MAIN_FOLDER) if os.path.isdir(os.path.join(MAIN_FOLDER, fol)) ]

article_file_paths = [os.path.join(fol, file) for fol in article_folders for file in os.listdir(fol) ]

print("we have", len([x for x in article_file_paths if "html" in x]), "article HTMLs to scrape")

print("\n\nhere are the first few filenames:\n")
pp(article_file_paths[:5])
print(os.path.split("stanford.library.sydney.edu.au/entries/meaning/index.html"))
print(os.path.split(os.path.split("stanford.library.sydney.edu.au/entries/meaning/index.html")[0]))
df = pd.DataFrame(article_file_paths)

df.columns = ["filenames"]

def get_filetype(x):
    return os.path.splitext(x)[-1]


def get_topic(x):
    path, fil = os.path.split(x)
    _, fol = os.path.split(path)
    return fol



df['filetype'] = df['filenames'].apply(get_filetype)
df['topic'] = df['filenames'].apply(get_topic)

df.head()
class MESSAGE(Enum):
    SUCCESS = {'num': 0, 'msg': "all good"}
    ARTICLE_ERROR = {'num': 1, 'msg': "expected exactly 1 id='article'"}
    MAIN_TEXT_ERROR = {'num': 2, 'msg': "expected there to be an id='main-text'"}
    FAILURE = {'num': 3, 'msg': "something bad"}
    def __bool__(self):
        """
        >>> MESSAGE.SUCCESS or MESSAGE.FAILURE
        <<< MESSAGE.FAILURE
        """
        return self.value['num'] != 0
    def choose_over(self, older_error):
        """
        prefer latest errors over older errors
        prefer errors over successes
        prefer specific errors over general errors
        """
        if self is not self.SUCCESS:  
            # prefer latest errors over older errors
            if self is self.FAILURE:
                return older_error or self
            return self
        else:
            return older_error
print(MESSAGE.ARTICLE_ERROR.choose_over(MESSAGE.SUCCESS))  # prefer errors over successes
print(MESSAGE.SUCCESS.choose_over(MESSAGE.ARTICLE_ERROR))  # prefer errors over successes
print(MESSAGE.MAIN_TEXT_ERROR.choose_over(MESSAGE.ARTICLE_ERROR))  # prefer latest errors over older errors
print(MESSAGE.ARTICLE_ERROR.choose_over(MESSAGE.MAIN_TEXT_ERROR))  # prefer latest errors over older errors
print(MESSAGE.FAILURE.choose_over(MESSAGE.ARTICLE_ERROR))  # prefer specific errors over general errors
print(MESSAGE.ARTICLE_ERROR.choose_over(MESSAGE.FAILURE))  # prefer specific errors over general errors
print(MESSAGE.SUCCESS or MESSAGE.FAILURE)
print(MESSAGE.FAILURE or MESSAGE.SUCCESS)
print(MESSAGE.FAILURE or MESSAGE.ARTICLE_ERROR)
print(MESSAGE.ARTICLE_ERROR or MESSAGE.FAILURE)
DEBUG = False


def get_soup(filename):
    with open(filename, 'r') as f:
        print(filename) if DEBUG else None
        return BeautifulSoup(f, 'html.parser')


def is_heading(sib):
    return (
        isinstance(sib, Tag) 
        and 
        sib.name in ['h1','h2','h3','h4','h5','h6']
    )
    
    

def _get_metadata(soup):
    metadata_list = soup.find_all('meta')
    doc_title = soup.find('meta', attrs={'name':'DC.title'})
    doc_creator = soup.find('meta', attrs={'name':'DC.creator'})
    doc_terms_modified = soup.find('meta', attrs={'name':'DCTERMS.modified'})
    citation_pub_date = soup.find('meta', attrs={'property':'citation_publication_date'})
    citation_author = soup.find('meta', attrs={'property':'citation_author'})
    return {
        "doc_title_html": doc_title,
        "doc_creator_html": doc_creator,
        "doc_terms_modified_html": doc_terms_modified,
        "citation_pub_date_html": citation_pub_date,
        "citation_author_html": citation_author,
    }


def _get_article_content(article=None):
    if article == None:
        return None

    article_content_list = article.find_all("div", id="article-content")
    # ASSUME: there is only one article && HTML structure has 'article-content'
    assert len(article_content_list) == 1
    article_content = article_content_list[0]
    return article_content


def _get_article(soup):
    article_list = soup.find_all("div", id="article")
    # ASSUME: there is only one article
    if len(article_list) != 1:
        print(f"WEIRD! {MESSAGE.ARTICLE_ERROR.value}") if DEBUG else None
        return (None, MESSAGE.ARTICLE_ERROR)
    article = article_list[0]
    return (article, MESSAGE.SUCCESS)


def _get_user_section(article_content=None):
    if article_content == None:
        return None
    article_user_editable_section_list = article_content.find_all("div", id="aueditable")
    # ASSUME: there is only one article && HTML structure has 'article-content' && "aueditable" in structure too
    assert len(article_user_editable_section_list) == 1
    user_section = article_user_editable_section_list[0]
    return user_section


def _get_structured_content_data(root_tag=None):
    err_msg = MESSAGE.SUCCESS
    if root_tag == None:
        return (
            {
                "pubinfo_html"         : None, 
                "preamble_html"        : None, 
                "toc_html"             : None, 
                "main_text_html"       : None, 
                "biblio_html"          : None, 
                "academic_tools_html"  : None, 
                "resources_html"       : None, 
                "related_entries_html" : None,
            },
            MESSAGE.FAILURE
        )
    
    pubinfo_list         = root_tag.find_all("div", id="pubinfo") or None
    preamble_list        = root_tag.find_all("div", id="preamble") or None
    toc_list             = root_tag.find_all("div", id="toc") or None
    main_text_list       = root_tag.find_all("div", id="main-text") or None
    biblio_list          = root_tag.find_all("div", id="bibliography") or None
    academic_tools_list  = root_tag.find_all("div", id="academic-tools") or None
    resources_list       = root_tag.find_all("div", id="other-internet-resources") or None
    related_entries_list = root_tag.find_all("div", id="related-entries") or None

    div_lists = [
        pubinfo_list,
        preamble_list,
        toc_list,
        main_text_list,
        biblio_list,
        academic_tools_list,
        resources_list,
        related_entries_list,
    ]
    # ASSUME: there is a main_text
    if main_text_list == None:
        print(f"VERY WEIRD! {MESSAGE.MAIN_TEXT_ERROR.value}") if DEBUG else None
        err_msg = MESSAGE.MAIN_TEXT_ERROR

    return (
        {
            "pubinfo_html"         : pubinfo_list[0] if pubinfo_list else None,
            "preamble_html"        : preamble_list[0] if pubinfo_list else None,
            "toc_html"             : toc_list[0] if pubinfo_list else None,
            "main_text_html"       : main_text_list[0] if pubinfo_list else None,
            "biblio_html"          : biblio_list[0] if pubinfo_list else None,
            "academic_tools_html"  : academic_tools_list[0] if pubinfo_list else None,
            "resources_html"       : resources_list[0] if pubinfo_list else None,
            "related_entries_html" : related_entries_list[0] if pubinfo_list else None,
        },
        err_msg,
    )


def _get_sections_between_headings(main_text=None):
    if main_text == None:
        return None
    
    headings = main_text.find_all(re.compile('^h[1-6]$'))

    sections = []

    tag = headings[0]
    heading_text = headings[0].text
    section_num = 1
    sections.append(
        {
            "id": section_num,
            "heading_text": heading_text,
            "soup_data": str(tag),
        }
    )

    for sib in tag.next_siblings:
        if( is_heading(sib) ):
            section_num += 1
            heading_text = sib.text
            sections.append(
                {
                    "id": section_num,
                    "heading_text": heading_text,
                    "soup_data": str(sib),
                }
            )
        else:
            sections.append(
                {
                    "id": section_num,
                    "heading_text": heading_text,
                    "soup_data": str(sib),
                }
            )

    return sections


def _get_all_the_useful_data(soup):
    # this is good data
    metadata_dict = _get_metadata(soup)

    # drill into article data
    article, err_msg = _get_article(soup)
    article_content = _get_article_content(article)
    user_section = _get_user_section(article_content)

    root_tag = user_section

    struct_data_dict, err_msg2 = _get_structured_content_data(root_tag)
    
    final_err_msg = err_msg2.choose_over(err_msg)
    
    main_text = struct_data_dict['main_text_html']

    # the main thing we care about
    sections = _get_sections_between_headings(main_text)

    return (
        { 
            "sections": sections, 
            "article_html": article,
            "article_content_html": article_content,
            "user_section_html": user_section,
            **struct_data_dict, 
            **metadata_dict
        },
        final_err_msg
    )
    

def get_full_html_and_plain_text_and_sections(filename):
    soup = get_soup(filename)
    html = soup.html
    plain_text = soup.text
    useful_data, err_msg = _get_all_the_useful_data(soup)
    data = {
        "full_html": html,
        "plain_text": plain_text,
        "err_msg": err_msg,
        **useful_data
    }
    return pd.Series(data)
# SAMPLE_FILE = "stanford.library.sydney.edu.au/entries/meaning/index.html"
# SAMPLE_SOUP = get_soup(SAMPLE_FILE)

# SAMPLE_METADATA_list = SAMPLE_SOUP.find_all('meta')
# print("\n", SAMPLE_METADATA_list, "\n")
# SAMPLE_DOC_TITLE = SAMPLE_SOUP.find('meta', attrs={'name':'DC.title'})
# SAMPLE_DOC_CREATOR = SAMPLE_SOUP.find('meta', attrs={'name':'DC.creator'})
# SAMPLE_DOC_TERMS_MODIFIED = SAMPLE_SOUP.find('meta', attrs={'name':'DCTERMS.modified'})
# SAMPLE_DOC_PUB_DATE = SAMPLE_SOUP.find('meta', attrs={'property':'citation_publication_date'})
# SAMPLE_DOC_AUTHOR = SAMPLE_SOUP.find('meta', attrs={'property':'citation_author'})

# SAMPLE_ARTICLES = SAMPLE_SOUP.find_all("div", id="article")
# print("there are", len(SAMPLE_ARTICLES), "articles within `SAMPLE_SOUP`")
# # ASSUME: there is only one article
# assert len(SAMPLE_ARTICLES) == 1
# SAMPLE_ARTICLE = SAMPLE_ARTICLES[0]


# SAMPLE_ARTICLE_CONTENTS = SAMPLE_ARTICLE.find_all("div", id="article-content")
# print("there are", len(SAMPLE_ARTICLE_CONTENTS), "top-level article containers within `SAMPLE_ARTICLE`")
# # ASSUME: there is only one article && HTML structure has 'article-content'
# assert len(SAMPLE_ARTICLE_CONTENTS) == 1
# SAMPLE_ARTICLE_CONTENT = SAMPLE_ARTICLE_CONTENTS[0]


# SAMPLE_ARTICLE_USER_EDITABLE_SECTIONS = SAMPLE_ARTICLE_CONTENT.find_all("div", id="aueditable")
# print("there are", len(SAMPLE_ARTICLE_USER_EDITABLE_SECTIONS), "user editable sections within `SAMPLE_ARTICLE_CONTENT`")
# # ASSUME: there is only one article && HTML structure has 'article-content' && "aueditable" in structure too
# assert len(SAMPLE_ARTICLE_USER_EDITABLE_SECTIONS) == 1
# SAMPLE_USER_SECTION = SAMPLE_ARTICLE_USER_EDITABLE_SECTIONS[0]


# SAMPLE_PUBINFO_list                   = SAMPLE_USER_SECTION.find_all("div", id="pubinfo")
# SAMPLE_PREAMBLE_list                  = SAMPLE_USER_SECTION.find_all("div", id="preamble")
# SAMPLE_TOC_list                       = SAMPLE_USER_SECTION.find_all("div", id="toc")
# SAMPLE_MAIN_TEXT_list                 = SAMPLE_USER_SECTION.find_all("div", id="main-text")
# SAMPLE_BIBLIO_list                    = SAMPLE_USER_SECTION.find_all("div", id="bibliography")
# SAMPLE_ACADEMIC_TOOLS_list            = SAMPLE_USER_SECTION.find_all("div", id="academic-tools")
# SAMPLE_ONLINE_INTERNET_RESOURCES_list = SAMPLE_USER_SECTION.find_all("div", id="other-internet-resources")
# SAMPLE_RELATED_ENTRIES_list           = SAMPLE_USER_SECTION.find_all("div", id="related-entries")

# div_lists = [
#     SAMPLE_PUBINFO_list,
#     SAMPLE_PREAMBLE_list,
#     SAMPLE_TOC_list,
#     SAMPLE_MAIN_TEXT_list,
#     SAMPLE_BIBLIO_list,
#     SAMPLE_ACADEMIC_TOOLS_list,
#     SAMPLE_ONLINE_INTERNET_RESOURCES_list,
#     SAMPLE_RELATED_ENTRIES_list
# ]
# # ASSUME: all of this HTML structure
# assert all([len(x) == 1 for x in div_lists])

# SAMPLE_PUBINFO                   = SAMPLE_PUBINFO_list[0]
# SAMPLE_PREAMBLE                  = SAMPLE_PREAMBLE_list[0]
# SAMPLE_TOC                       = SAMPLE_TOC_list[0]
# SAMPLE_MAIN_TEXT                 = SAMPLE_MAIN_TEXT_list[0]
# SAMPLE_BIBLIO                    = SAMPLE_BIBLIO_list[0]
# SAMPLE_ACADEMIC_TOOLS            = SAMPLE_ACADEMIC_TOOLS_list[0]
# SAMPLE_ONLINE_INTERNET_RESOURCES = SAMPLE_ONLINE_INTERNET_RESOURCES_list[0]
# SAMPLE_RELATED_ENTRIES           = SAMPLE_RELATED_ENTRIES_list[0]

# SAMPLE_SECTIONS = _get_sections_between_headings(SAMPLE_MAIN_TEXT)

# print("there are", len(SAMPLE_SECTIONS), "sections within `SAMPLE_MAIN_TEXT`")
html_df = df[df['filetype'] == '.html']
series = html_df['filenames']
fun = get_full_html_and_plain_text_and_sections
mask = ['full_html', 'plain_text', 'err_msg', 'sections', 'article_html',
       'article_content_html', 'user_section_html', 'pubinfo_html',
       'preamble_html', 'toc_html', 'main_text_html', 'biblio_html',
       'academic_tools_html', 'resources_html', 'related_entries_html',
       'doc_title_html', 'doc_creator_html', 'doc_terms_modified_html',
       'citation_pub_date_html', 'citation_author_html']
df[mask] = series.apply(fun)
# df[mask] = series[:20].apply(fun)
x = get_full_html_and_plain_text_and_sections('stanford.library.sydney.edu.au/entries/contractarianism/index.html')
x['sections']

get_full_html_and_plain_text_and_sections("stanford.library.sydney.edu.au/entries/albo-joseph/vita.html")
def _get_related_entries_list(soup):
    return [a['href'] for a in soup.find_all('a', href=True)] if isinstance(soup, Tag) else None

df['related_entries_list'] = df['related_entries_html'].apply(_get_related_entries_list)
def _get_preamble_text(soup):
    return soup.text if isinstance(soup, Tag) else None

df['preamble_text'] = df['preamble_html'].apply(_get_preamble_text)
def _get_author(soup):
    return soup['content'] if isinstance(soup, Tag) else None

df['author'] = df['citation_author_html'].apply(_get_author)
def _get_creator(soup):
    return soup['content'] if isinstance(soup, Tag) else None

df['creator'] = df['doc_creator_html'].apply(_get_creator)
def _get_title(soup):
    return soup['content'] if isinstance(soup, Tag) else None

df['title'] = df['doc_title_html'].apply(_get_title)
def soup_data_agg(series):
    #str_series = [str(x) for x in series]
    agg_str = "".join(series)
    doc = BeautifulSoup(agg_str)
    paragraph_list = doc.find_all('p')
    iteration = zip(paragraph_list, range(1,len(paragraph_list)+1))
    p_struct_list = [{"id": i, "text": p.text} for p, i in iteration]
    if p_struct_list != []:
        return p_struct_list
    elif paragraph_list != []:
        text = "".join([x.text for x in paragraph_list])
        return [{"id": 1, "text": text}]
    else:
        return None
    

def heading_text_agg(series):
    s = set(series)
    res = s.pop()
    assert s == set(), f"should only be 1 heading_text.. but s = {s}"
    return res


agg_funs = {
    "heading_text": heading_text_agg,
    "soup_data": soup_data_agg,
}

def paragraph_agg(series):
    return "".join(series)


def organize_paragraphs(paragraph_dict_list):
    if type(paragraph_dict_list) == list:
        sdf = pd.DataFrame(paragraph_dict_list)
        agg = sdf.groupby('id').agg(agg_funs)#.rename(columns={'text': 'paragraphs'})
        return "text"
    else:
        return None    


def organize_section_data(section_dict_list):
    if type(section_dict_list) == list:
        sdf = pd.DataFrame(section_dict_list)
        agged = sdf.groupby('id').agg(agg_funs).rename(columns={'soup_data': 'paragraphs'})
        return list(agged.reset_index().T.to_dict().values())
    else:
        return None

df['sections_agged'] = df['sections'].apply(organize_section_data)

#list_dict_data = df['sections'].values[1]
#sections_df_1 = pd.DataFrame(list_dict_data)
#sections_df_1_agged = sections_df_1.groupby('id').agg(agg_funs).rename(columns={'soup_data': 'paragraphs'})
#ps_df_0 = pd.DataFrame(sections_df_1_agged['paragraphs'].values[0])
#ps_df.groupby('id').agg(paragraph_agg).rename(columns={'text': 'paragraph'})
df.columns
df['sections_agged'][2]
df.head()
simple_df = df[ ['filenames', 'filetype', 'topic', 'title', 
                 'author', 'creator', 'preamble_text', 
                 'sections_agged', 'related_entries_list', 
                 'plain_text'] 
              ]

simple_df = simple_df.rename(columns={"sections_agged": "sections"})
simple_df.head()
CSV_DATA = df.to_csv(index=False)
CSV_FILENAME = "data.csv"

with open(CSV_FILENAME, 'w') as f:
    f.write(CSV_DATA)
del df
del CSV_DATA
del f
gc.collect()
SIMPLE_CSV_DATA = simple_df.to_csv(index=False)
SIMPLE_CSV_FILENAME = "simple_data.csv"

with open(SIMPLE_CSV_FILENAME, 'w') as f:
    f.write(SIMPLE_CSV_DATA)
del simple_df
del SIMPLE_CSV_DATA
del f
gc.collect()
!ls
df = pd.read_csv(CSV_FILENAME)
df.head()
# %timeit -n1 pd.read_csv(CSV_FILENAME).head()
print("60 ms ± 4.29 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)")
df['filetype'].astype('category').dtype
df['topic'].astype('category').dtype.categories.tolist()
!rm -rf stanford.library.sydney.edu.au
!ls
df.explode("related_entries_list").head()
# df.explore("sections_agged").head()
