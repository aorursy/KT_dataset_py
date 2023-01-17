import PyPDF2 as pdf
file = open("filename.pdf", 'rb')
pdf_reader = pdf.PdfFileReader(file)
pdf_reader.getIsEncrypted() # returns if the pdf is encrypted or not
pdf_reader.getNumPages() # returns number of pages in the pdf

for i in range(pdf_reader.getNumPages()):
    page = pdf_reader.getPage(i)
    print(page.extractText())
