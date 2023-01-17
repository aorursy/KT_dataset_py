!pip install pdfminer
!pip install cStringIO
from pdfminer.pdfinterp import PDFResourceManager,PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.pdfpage import PDFPage
from pdfminer.layout import LAParams
from io import StringIO 
import pdfminer
def hey_pdf(path):
    manager = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    layout = LAParams(all_texts=True)
    device = TextConverter(manager,retstr,laparams=layout)
    filepath=open(path,'rb')
    interpreter = PDFPageInterpreter(manager,device)
    for page in PDFPage.get_pages(filepath, check_extractable= True):
        interpreter.process_page(page)
        text = retstr.getvalue()
    filepath.close()
    device.close()
    retstr.close()
    return text
print(hey_pdf("../input/researchpaper/Researh_Paper_version2.pdf"))