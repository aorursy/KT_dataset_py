# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# Any results you write to the current directory are saved as output.
from pdfminer.pdfparser import PDFParser

from pdfminer.pdfdocument import PDFDocument

from pdfminer.pdfpage import PDFPage

from pdfminer.pdfpage import PDFTextExtractionNotAllowed

from pdfminer.pdfinterp import PDFResourceManager

from pdfminer.pdfinterp import PDFPageInterpreter

from pdfminer.pdfdevice import PDFDevice



# Open a PDF file.

fp = open("/kaggle/input/declaratieformulier-hollandzorg-du.pdf", 'rb')

# Create a PDF parser object associated with the file object.

parser = PDFParser(fp)

# Create a PDF document object that stores the document structure.

# Supply the password for initialization.

document = PDFDocument(parser)

# Check if the document allows text extraction. If not, abort.

if not document.is_extractable:

    raise PDFTextExtractionNotAllowed

# Create a PDF resource manager object that stores shared resources.

rsrcmgr = PDFResourceManager()

# Create a PDF device object.

device = PDFDevice(rsrcmgr)

# Create a PDF interpreter object.

interpreter = PDFPageInterpreter(rsrcmgr, device)

# Process each page contained in the document.

for page in PDFPage.create_pages(document):

    print(interpreter.process_page(page))

    
