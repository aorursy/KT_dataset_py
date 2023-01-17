from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument
from pdfminer.pdftypes import resolve1
import json
dict={}
dict1={}

fp = open('filename.pdf', 'rb')

parser = PDFParser(fp)
doc = PDFDocument(parser)
fields = resolve1(doc.catalog['AcroForm'])['Fields']
for i in fields:
    field = resolve1(i)
    name, value = field.get('T'), field.get('V')
    dict[format(name)]=format(value)
print(dict)

for key, value in dict.items():
    dict1[key[2:-1]]=value[2:-1]
print(dict1)

json_data=json.dumps(dict1) # returns field name and field value in Key-Value pairs of JSON Format 
print(json_data)
