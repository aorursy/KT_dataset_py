from lxml import etree
from whoosh.fields import Schema, TEXT, ID,NUMERIC
from whoosh.index import create_in,open_dir
from whoosh.qparser import QueryParser
import shutil
import os.path

### Whoosh ini

class TitleTarget(object):
    def __init__(self,writer):
        self.text = []
        self.name = ""
        self.cs = ""
        self.se = ""
        self.writer = writer
        

    def start(self, tag, attrib):
        self.is_name = True if tag == "{http://www.drugbank.ca}name" else False
        self.is_indication = True if tag == "{http://www.drugbank.ca}indication" else False
        self.is_toxicity = True if tag == "{http://www.drugbank.ca}toxicity" else False
    def end(self, tag):
        if tag == "{http://www.drugbank.ca}drug":
          
            self.writer.add_document(name=self.name, cs=self.cs,se=self.se)
            self.name=""
            self.cs=""
            self.se=""
    def data(self, data):
        if self.name=="" and self.is_name:
            self.name += str(data).replace("'b'",'')
        elif self.is_indication:
            self.cs += str(data).replace("'b'",'')
        elif self.is_toxicity :
            self.se += str(data).replace("'b'",'')
    def close(self):
        self.writer.commit()
        return

class Drugbank:
    """ Object used to work on the file "drugbank.xml"
    """

    def __init__(self):
        print("Opening Drugbank -> Start")
        self.schema: Schema = Schema(name= ID(stored=True), cs = TEXT(stored=True),se=TEXT(stored=True))
        if not os.path.exists("index_drugbank"):
            os.mkdir("index_drugbank")
            self.ix = create_in("index_drugbank",self.schema, "drugbank")
        self.ix = open_dir("index_drugbank","drugbank")
        self.writer = self.ix.writer()
        if self.ix.is_empty() :
            print("Indexing Drugbank -> Start")
            infile = './ressources/drugbank.xml'
            parser = etree.XMLParser(target = TitleTarget(self.writer))
            etree.parse(infile, parser)
            print("Indexing Drugbank -> End")    
        else :
            print("Delete index_drugbank if you want to reindexate DrugBank")
        self.searcher = self.ix.searcher()
        print("Opening Drugbank -> End\n")

    def search(self,cs:str,se):
        """Search for a clinical sign
        
        Arguments:
            cs {str} -- [clinical sign]
        
        Returns:
            [list] -- list of the name of drugs
        """ 
        if se==0:
            parser = QueryParser("cs",self.ix.schema)
        else:
            parser = QueryParser("se",self.ix.schema)
        myquery = parser.parse(cs)
        query = self.searcher.search(myquery, limit=None)
        results = []
        for r in query :
            results.append("DrugBank:"+r.get("name"))
        return results 
