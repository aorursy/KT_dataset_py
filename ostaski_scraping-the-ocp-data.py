import urllib.request

from bs4 import BeautifulSoup

import time

import os



# Captain Obvious reminds us to enable Internet (on Kaggle)

# let's download the Combined pages to see how we can isolate the data with BeautifulSoup

combined_ids = ['427','430','444','448','456','477','494']

base_url = 'http://ocpdb.pythonanywhere.com/ocpdb/'



# making headers to identify myself to the sysadmins

headers = {

    'User-Agent': 'Bill Ostaski, https://www.kaggle.com/ostaski/scraping-the-ocp-data',

    'From': 'ostaski@gmail.com'

}



# putting these in a "Pages" directory

dir = "Pages"

if not os.path.exists(dir):

    os.mkdir(dir)

    for id in combined_ids:

        url = base_url + id

        req = urllib.request.Request(url, headers=headers)

        resp = urllib.request.urlopen(req)

        with open("Pages/" + id, "a") as p:

            p.writelines(str(resp.read()))

        time.sleep(10) # showing some respect to the server

f = open("Pages/427", "r")

if f.mode == 'r':

    contents = f.read()

    soup = BeautifulSoup(contents, 'html.parser')

    print(soup.prettify()) # wish I could truncate this output
# we have some links in our data points, let's have a look

for link in soup.find_all('a'):

    print(link.get('href'))
soup.body.b.text
soup.find_all("td", class_="numcell")
# defining some functions below

def getReportType(tocpid):

    if tocpid in combined_ids:

        return 'B' # B for Both

    elif int(tocpid) < 675:

        return 'C' # C for Chemical

    else:

        return 'G' # G for Genetic
# function for "Both" report type files

def populateBoth(soup):

    ocpid = soup.find(class_="text-right").find('b').find(text=True, recursive=False).string[7:]

    strain = soup.body.b.text

    sampleID = soup.find(class_="text-muted").find(text=True, recursive=False)

    dateRecorded = soup.find(class_="text-right").find(text=True, recursive=False).string[1:]

    # get truncated ocpid

    tocpid = ocpid.lstrip("0")

    reportType = getReportType(tocpid)

    chemicalLab = soup.find('td', colspan="8").find(text=True, recursive=False)[2:]

    h20 = soup.find('td', colspan="4").text[6:]

    totalTHC = soup.find('td', colspan="3").text[11:]

    thc = soup.find_all("td", class_="numcell")[0].text

    # can't use hyphens in identifiers

    Δ8_thc = soup.find_all("td", class_="numcell")[3].text

    Δ9_thc = soup.find_all("td", class_="numcell")[6].text

    thca = soup.find_all("td", class_="numcell")[9].text

    thcv = soup.find_all("td", class_="numcell")[12].text

    totalCBD = soup.find("td", colspan="2").text[11:]

    cbda = soup.find_all("td", class_="numcell")[17].text

    cbdv = soup.find_all("td", class_="numcell")[20].text

    cbdva = soup.find_all("td", class_="numcell")[23].text

    cbc = soup.find_all("td", class_="numcell")[26].text

    cbg = soup.find_all("td", class_="numcell")[29].text

    cbn = soup.find_all("td", class_="numcell")[31].text

    α_pinene = soup.find_all("td", class_="numcell")[1].text

    camphene = soup.find_all("td", class_="numcell")[4].text

    myrcene = soup.find_all("td", class_="numcell")[7].text

    β_pinene = soup.find_all("td", class_="numcell")[10].text

    three_carene = soup.find_all("td", class_="numcell")[13].text

    α_terpinene = soup.find_all("td", class_="numcell")[15].text

    d_limonene = soup.find_all("td", class_="numcell")[18].text

    p_cymene = soup.find_all("td", class_="numcell")[21].text

    ocimene = soup.find_all("td", class_="numcell")[24].text

    eucalyptol = soup.find_all("td", class_="numcell")[27].text

    y_terpinene = soup.find_all("td", class_="numcell")[30].text

    terpinolene = soup.find_all("td", class_="numcell")[32].text

    linalool = soup.find_all("td", class_="numcell")[2].text

    isopulegol = soup.find_all("td", class_="numcell")[5].text

    geraniol = soup.find_all("td", class_="numcell")[8].text

    β_caryophyllene = soup.find_all("td", class_="numcell")[11].text

    α_humelene = soup.find_all("td", class_="numcell")[14].text

    nerolidol_1 = soup.find_all("td", class_="numcell")[16].text

    nerolidol_2 = soup.find_all("td", class_="numcell")[19].text

    guaiol = soup.find_all("td", class_="numcell")[22].text

    caryophylleneOxide = soup.find_all("td", class_="numcell")[25].text

    α_bisabolol = soup.find_all("td", class_="numcell")[28].text

    geneticLab = soup.find_all('td', colspan="5")[1].text[6:23]

    sample = soup.find_all('a')[8].text

    sampleURL = soup.find_all('a')[8]

    organism = soup.find_all('a')[9].text

    organismURL = soup.find_all('a')[9]

    project = soup.find_all('a')[10].text

    projectURL = soup.find_all('a')[10]

    study = soup.find_all('a')[11].text

    studyURL = soup.find_all('a')[11]

    run = soup.find_all('a')[12].text

    runURL = soup.find_all('a')[12]

    datePublished = soup.find_all("td", class_="numcell")[33].text

    spots = soup.find_all("td", class_="numcell")[34].text

    bases = soup.find_all("td", class_="numcell")[35].text

    size = soup.find_all("td", class_="numcell")[36].text.replace('\\xc2\\xa0', ' ')

    notes = soup.select('div.col')[3].text[23:27]

   

    return [ocpid,strain,sampleID,dateRecorded,reportType,chemicalLab,h20,totalTHC,thc,Δ8_thc,

            Δ9_thc,thca,thcv,totalCBD,cbda,cbdv,cbdva,cbc,cbg,cbn,α_pinene,camphene,myrcene,

            β_pinene,three_carene,α_terpinene,d_limonene,p_cymene,ocimene,eucalyptol,y_terpinene,

            terpinolene,linalool,isopulegol,geraniol,β_caryophyllene,α_humelene,nerolidol_1,

            nerolidol_2,guaiol,caryophylleneOxide,α_bisabolol,geneticLab,sample,sampleURL,organism,

            organismURL,project,projectURL,study,studyURL,run,runURL,datePublished,spots,bases,

            size,notes]
# function for "Chemical" report type files

def populateChemical(soup):

    ocpid = soup.find(class_="text-right").find('b').find(text=True, recursive=False).string[7:]

    strain = soup.body.b.text

    sampleID = soup.find(class_="text-muted").find(text=True, recursive=False)

    dateRecorded = soup.find(class_="text-right").find(text=True, recursive=False).string[1:]

    # get truncated ocpid

    tocpid = ocpid.lstrip("0")

    reportType = getReportType(tocpid)

    chemicalLab = soup.find('td', colspan="8").find(text=True, recursive=False)[2:]

    h20 = soup.find('td', colspan="4").text[6:]

    totalTHC = soup.find('td', colspan="3").text[11:]

    thc = soup.find_all("td", class_="numcell")[0].text

    # can't use hyphens in identifiers

    Δ8_thc = soup.find_all("td", class_="numcell")[3].text

    Δ9_thc = soup.find_all("td", class_="numcell")[6].text

    thca = soup.find_all("td", class_="numcell")[9].text

    thcv = soup.find_all("td", class_="numcell")[12].text

    totalCBD = soup.find("td", colspan="2").text[11:]

    cbda = soup.find_all("td", class_="numcell")[17].text

    cbdv = soup.find_all("td", class_="numcell")[20].text

    cbdva = soup.find_all("td", class_="numcell")[23].text

    cbc = soup.find_all("td", class_="numcell")[26].text

    cbg = soup.find_all("td", class_="numcell")[29].text

    cbn = soup.find_all("td", class_="numcell")[31].text

    α_pinene = soup.find_all("td", class_="numcell")[1].text

    camphene = soup.find_all("td", class_="numcell")[4].text

    myrcene = soup.find_all("td", class_="numcell")[7].text

    β_pinene = soup.find_all("td", class_="numcell")[10].text

    three_carene = soup.find_all("td", class_="numcell")[13].text

    α_terpinene = soup.find_all("td", class_="numcell")[15].text

    d_limonene = soup.find_all("td", class_="numcell")[18].text

    p_cymene = soup.find_all("td", class_="numcell")[21].text

    ocimene = soup.find_all("td", class_="numcell")[24].text

    eucalyptol = soup.find_all("td", class_="numcell")[27].text

    y_terpinene = soup.find_all("td", class_="numcell")[30].text

    terpinolene = soup.find_all("td", class_="numcell")[32].text

    linalool = soup.find_all("td", class_="numcell")[2].text

    isopulegol = soup.find_all("td", class_="numcell")[5].text

    geraniol = soup.find_all("td", class_="numcell")[8].text

    β_caryophyllene = soup.find_all("td", class_="numcell")[11].text

    α_humelene = soup.find_all("td", class_="numcell")[14].text

    nerolidol_1 = soup.find_all("td", class_="numcell")[16].text

    nerolidol_2 = soup.find_all("td", class_="numcell")[19].text

    guaiol = soup.find_all("td", class_="numcell")[22].text

    caryophylleneOxide = soup.find_all("td", class_="numcell")[25].text

    α_bisabolol = soup.find_all("td", class_="numcell")[28].text

    geneticLab = ''

    sample = ''

    sampleURL = ''

    organism = ''

    organismURL = ''

    project = ''

    projectURL = ''

    study = ''

    studyURL = ''

    run = ''

    runURL = ''

    datePublished = ''

    spots = ''

    bases = ''

    size = ''

    notes = soup.select('div.col')[3].text[23:27]

   

    return [ocpid,strain,sampleID,dateRecorded,reportType,chemicalLab,h20,totalTHC,thc,Δ8_thc,

            Δ9_thc,thca,thcv,totalCBD,cbda,cbdv,cbdva,cbc,cbg,cbn,α_pinene,camphene,myrcene,

            β_pinene,three_carene,α_terpinene,d_limonene,p_cymene,ocimene,eucalyptol,y_terpinene,

            terpinolene,linalool,isopulegol,geraniol,β_caryophyllene,α_humelene,nerolidol_1,

            nerolidol_2,guaiol,caryophylleneOxide,α_bisabolol,geneticLab,sample,sampleURL,organism,

            organismURL,project,projectURL,study,studyURL,run,runURL,datePublished,spots,bases,

            size,notes]
# function for "Genetic" report type files

def populateGenetic(soup):

    ocpid = soup.find(class_="text-right").find('b').find(text=True, recursive=False).string[7:]

    strain = soup.body.b.text.replace('\\n\\t\\t\\t ', '').replace('\\n\\t\\t\\t  ', '').strip()

    sampleID = soup.find(class_="text-muted").find(text=True, recursive=False)

    dateRecorded = soup.find(class_="text-right").find(text=True, recursive=False).string[1:]

    # get truncated ocpid

    tocpid = ocpid.lstrip("0")

    reportType = getReportType(tocpid)

    chemicalLab = ''

    h20 = ''

    totalTHC = ''

    thc = ''

    Δ8_thc = ''

    Δ9_thc = ''

    thca = ''

    thcv = ''

    totalCBD = ''

    cbda = ''

    cbdv = ''

    cbdva = ''

    cbc = ''

    cbg = ''

    cbn = ''

    α_pinene = ''

    camphene = ''

    myrcene = ''

    β_pinene = ''

    three_carene = ''

    α_terpinene = ''

    d_limonene = ''

    p_cymene = ''

    ocimene = ''

    eucalyptol = ''

    y_terpinene = ''

    terpinolene = ''

    linalool = ''

    isopulegol = ''

    geraniol = ''

    β_caryophyllene = ''

    α_humelene = ''

    nerolidol_1 = ''

    nerolidol_2 = ''

    guaiol = ''

    caryophylleneOxide = ''

    α_bisabolol = ''

    geneticLab = soup.find('td', colspan="8").find(text=True, recursive=False)[2:]

    sample = soup.find_all('a')[8].text

    sampleURL = soup.find_all('a')[8]

    organism = soup.find_all('a')[9].text

    organismURL = soup.find_all('a')[9]

    project = soup.find_all('a')[10].text

    projectURL = soup.find_all('a')[10]

    study = soup.find_all('a')[11].text

    studyURL = soup.find_all('a')[11]

    run = soup.find_all('a')[12].text

    runURL = soup.find_all('a')[12]

    datePublished = soup.find_all('td', colspan="5")[2].text[17:]

    spots = soup.find_all('td', colspan="5")[3].text[8:]

    bases = soup.find_all('td', colspan="5")[4].text[8:]

    size = soup.find_all('td', colspan="5")[5].text[7:].replace('\\xc2\\xa0', ' ')

    notes = ''

   

    return [ocpid,strain,sampleID,dateRecorded,reportType,chemicalLab,h20,totalTHC,thc,Δ8_thc,

            Δ9_thc,thca,thcv,totalCBD,cbda,cbdv,cbdva,cbc,cbg,cbn,α_pinene,camphene,myrcene,

            β_pinene,three_carene,α_terpinene,d_limonene,p_cymene,ocimene,eucalyptol,y_terpinene,

            terpinolene,linalool,isopulegol,geraniol,β_caryophyllene,α_humelene,nerolidol_1,

            nerolidol_2,guaiol,caryophylleneOxide,α_bisabolol,geneticLab,sample,sampleURL,organism,

            organismURL,project,projectURL,study,studyURL,run,runURL,datePublished,spots,bases,

            size,notes]
import requests

from bs4 import BeautifulSoup

import time

import datetime

import csv



combined_ids = ['427','430','444','448','456','477','494']



base_url = 'http://ocpdb.pythonanywhere.com/ocpdb/'



# making headers to identify myself to the sysadmin(s)

req_headers = {

    'User-Agent': 'Bill Ostaski, https://www.kaggle.com/ostaski/scraping-the-ocp-data',

    'From': 'ostaski@gmail.com'

}



# generally a good idea to note the date of this snapshot

filename = "OCPDB-" + datetime.datetime.now().strftime("%m_%d_%Y") + ".csv"



col_headers = ["OCPID","Strain","SampleID","DateRecorded","ReportType","ChemicalLab","H2O",

               "TotalTHC","THC","Δ8-THC","Δ9-THC","THCA","THCV","TotalCBD","CBDA","CBDV",

               "CBDVA","CBC","CBG","CBN","α-Pinene","Camphene","Myrcene","β-Pinene","3-Carene",

               "α-Terpinene","D-Limonene","p-Cymene","Ocimene","Eucalyptol","γ-Terpinene",

               "Terpinolene","Linalool","Isopulegol","Geraniol","β-Caryophyllene","α-Humelene",

               "Nerolidol-1","Nerolidol-2","Guaiol","CaryophylleneOxide","α-Bisabolol",

               "GeneticLab","Sample","SampleURL","Organism","OrganismURL","Project","ProjectURL",

               "Study","StudyURL","Run","RunURL","DatePublished","Spots","Bases","Size","Notes"]



# had to comment out the lines below in order to Commit

#with open(filename, "w+") as f:

#    writer = csv.writer(f)

#    writer.writerow(col_headers)



#for id in range(420, 1519): # only gets chemical report for combined ids

##for id in combined_ids: # grabs combined reports

#    url = base_url + str(id)

#    resp = requests.get(url, headers=req_headers)



#    soup = BeautifulSoup(resp.text, 'html.parser')

    

#    if id in combined_ids:

#        parsed_data = populateBoth(soup)

#    elif id < 675:

#        parsed_data = populateChemical(soup)

#    else:

#        parsed_data = populateGenetic(soup)

    

#    with open(filename, 'a') as f:

#        writer = csv.writer(f)

#        writer.writerow(parsed_data)



#    time.sleep(10) # showing some respect to the server