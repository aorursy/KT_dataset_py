!pip3 install -U scidownl
!scidownl -c 3
from scidownl.scihub import *

with open('../input/dois-batch-1/dois_1.txt',"r") as f:
    dois = f.readlines()
    for doi in dois:
        SciHub(doi, "out").download(choose_scihub_url_index=3)
        
from bs4 import BeautifulSoup
import os.path
import glob
tei_doc = '../input/tei-xml-files/tei_xml_files/0001.tei.xml'
#with open(tei_doc, 'r') as tei:
#    soup = BeautifulSoup(tei, 'lxml')
def read_tei(tei_doc):
    with open(tei_doc, 'r') as tei:
        soup = BeautifulSoup(tei, 'lxml')
        return soup
    raise RuntimeError('Cannot generate a soup from the input') 
def elem_to_text(elem, default=''):
    if elem:
        return elem.getText()
    else:
        return default
elem_to_text(soup.foobarelem, default="NA")
idno_elem = soup.find('idno', type='doi')
print(f"The doi is {idno_elem.getText()}")
from dataclasses import dataclass

@dataclass
class Person:
    firstname: str
    middlename: str
    surname: str

turing_author = Person(firstname='Alan', middlename='M', surname='Turing')

f"{turing_author.firstname} {turing_author.surname} authored many influential publications in computer science."
class TEIFile(object):
    def __init__(self, filename):
        self.filename = filename
        self.soup = read_tei(filename)
        self._text = None
        self._title = ''
        self._abstract = ''

    @property
    def doi(self):
        idno_elem = self.soup.find('idno', type='DOI')
        if not idno_elem:
            return ''
        else:
            return idno_elem.getText()

    @property
    def title(self):
        if not self._title:
            self._title = self.soup.title.getText()
        return self._title

    @property
    def abstract(self):
        if not self._abstract:
            abstract = self.soup.abstract.getText(separator=' ', strip=True)
            self._abstract = abstract
        return self._abstract

    @property
    def authors(self):
        authors_in_header = self.soup.analytic.find_all('author')

        result = []
        for author in authors_in_header:
            persname = author.persname
            if not persname:
                continue
            firstname = elem_to_text(persname.find("forename", type="first"))
            middlename = elem_to_text(persname.find("forename", type="middle"))
            surname = elem_to_text(persname.surname)
            person = Person(firstname, middlename, surname)
            result.append(person)
        return result
    
    @property
    def text(self):
        if not self._text:
            divs_text = []
            for div in self.soup.body.find_all("div"):
                # div is neither an appendix nor references, just plain text.
                if not div.get("type"):
                    div_text = div.get_text(separator=' ', strip=True)
                    divs_text.append(div_text)

            plain_text = " ".join(divs_text)
            self._text = plain_text
        return self._text
tei = TEIFile("../input/tei-xml-files/tei_xml_files/0001.tei.xml")
f"The authors of the paper entitled '{tei.title}' are {tei.authors}"
tei.abstract
from os.path import basename, splitext

def basename_without_ext(path):
    base_name = basename(path)
    stem, ext = splitext(base_name)
    if stem.endswith('.tei'):
        # Return base name without tei file
        return stem[0:-4]
    else:
        return stem
    
basename_without_ext(tei_doc)
def tei_to_csv_entry(tei_file):
    tei = TEIFile(tei_file)
    print(f"Handled {tei_file}")
    base_name = basename_without_ext(tei_file)
    return base_name, tei.doi, tei.title, tei.abstract, tei.authors, tei.text
tei_to_csv_entry(tei_doc)
import glob
from pathlib import Path

papers = sorted(Path("../input/tei-xml-files/tei_xml_files").glob('*.tei.xml'))
import multiprocessing
print(f"My machine has {multiprocessing.cpu_count()} cores.")

from multiprocessing.pool import Pool
pool = Pool()
csv_entries = pool.map(tei_to_csv_entry, papers)
csv_entries
import pandas as pd

result_csv = pd.DataFrame(csv_entries, columns=['paper_id', 'doi','title', 'abstract','authors','full_text'])
result_csv
result_csv.to_csv("metadata.csv", index=False)