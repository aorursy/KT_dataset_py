# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)



# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



import os

print(os.listdir("../input"))

import glob





# Any results you write to the current directory are saved as output.


bulletin_dir = "../input/cityofla/CityofLA/Job Bulletins"

directory = glob.glob(bulletin_dir + '/*')

fileList = [element.split("/")[-1] for element in directory]



print(fileList)
class Reader():

    def __init__(self, path, filename):

        self.path = path

        self.filename = filename

        

    def getFilename(self):

        spath = self.path + "/" + self.filename

        with open (spath, "r") as file1:

            fileList = [line for line in file1]

        print (fileList)

            

    def read(self):

        import datetime

        from datetime import date

        from itertools import islice

        fpath = self.path + "/" + self.filename

        with open (fpath, "r",encoding="latin-1") as file:

#------------------------------------------JOB_CLASS_TITLE---JOB_CLASS_NO--------------------------------------------------------------------------            

            self.jobTitle = ""

            self.jobNo = ""

            count = 0

            for line in file:

                count += 1

                if count == 1:

                    self.jobTitle = line.strip()

                    while self.jobTitle.isspace() or self.jobTitle == "":

                        self.jobTitle = list(islice(file, 1))[-1].strip()

                    

                    

                if "Class Code:" in self.jobTitle:

                    self.jobTitle = self.jobTitle.split("Class")[0].strip()

                        

                if ("Class Code:" in line or "Class  Code:" in line) and self.jobNo == "":

                    self.jobNo = line.split("Code:")[1].strip()

                    if len(self.jobNo) > 4:

                        self.jobNo = self.jobNo[0:4]

                                    

#------------------------------------------REQUIREMENT_SETS_AND_SUBSETS--------------------------------------------------------------------------                    

        with open (fpath, "r",encoding="latin-1") as file:            

            self.requirement = []

            self.requirementString = ""

            for line in file:

                if "REQUIREMENT" in line:

                    self.requirement.append(list(islice(file, 1))[-1].strip().replace("\t"," "))

                    #Debug exemptions with two blank lines instead of one

                    while self.requirement[0] == "":

                        self.requirement[0] = list(islice(file, 1))[-1].strip().replace("\t"," ")

                    while self.requirement[-1] != "":

                        self.requirement.append(list(islice(file, 1))[-1].strip().replace("\t"," "))

                        

                    #Delete last empty element of Array    

                    self.requirement = self.requirement[:-1]

                    

                    #Write conjunctions in captital letters

                    for i in range (0,len(self.requirement)):

                        if self.requirement[i][-2:] == "or":

                            self.requirement[i] = self.requirement[i][0:-2] + "####OR####"

                        if self.requirement[i][-3:] == "and":

                            self.requirement[i] = self.requirement[i][0:-3] + "####AND####"

                    

                    #Create string variable from Array         

                    self.requirementString = " ".join(self.requirement)

            

            #Debug inclusion of lines outside Requirements

            for i in range(0, len(self.requirement)):

                if "DUTIES" in self.requirement[i] or "NOTE" in self.requirement[i]:

                    del self.requirement[i:]

                    break

            

            #Creation of first Set + Subset

            self.requirementString1 = ""

            self.requirementSub1 = []

            subBegin = ["a.","b.","c.","d.","e.","f.","g.","h.","i.","j.","k.","l."]

            if len(self.requirement)>=1:

                self.requirementString1 = self.requirement[0]

                c1=0

                if len(self.requirement)>=2 and "2" != self.requirement[1][:1]: 

                    for i in range (1,len(self.requirement)):

                        if subBegin[i-1] == self.requirement[i][:2]:

                            self.requirementSub1.append(self.requirement[i])

                        elif "." not in self.requirement[i][:4]:

                            c1+=1

                            self.requirementSub1.append(subBegin[i-c1-1] + str(c1) + ") " + self.requirement[i])

                        elif c1!=0 and subBegin[i-1-c1] == self.requirement[i][:2]:

                            self.requirementSub1.append(self.requirement[i])

                    self.requirementString1 += " " + " ".join(self.requirementSub1)

                    



                        

            

            #Creation of further Sets and Subsets (pos = number of Set, Sub = 1 for Subset/= 0 for Set)

            def createSet(pos, sub):

                a = ""

                b = []

                c = 0

                d = 0

                if len(self.requirement) >= pos:

                    for i in range(pos-1,len(self.requirement)):

                        if str(pos) in self.requirement[i][:2]:

                            a = self.requirement[i]

                            c = i

                    if c != 0 and len(self.requirement) >= c:

                        for i in range (c,len(self.requirement)):

                            if str(pos+1) in self.requirement[i][:2]:

                                break

                            if subBegin[i-c] in self.requirement[i][:6] or subBegin[i-c][0]+")" in self.requirement[i][:6]:

                                    b.append(self.requirement[i])

                            elif subBegin[i-c-1] in self.requirement[i][:6] or subBegin[i-c-1][0]+")" in self.requirement[i][:6]:

                                    b.append(self.requirement[i])

                            elif "." not in self.requirement[i][:4] and str(pos) not in self.requirement[i][:2]:

                                    d += 1

                                    b.append("* " + self.requirement[i])

                            elif d !=0 and subBegin[i-c-d-1] == self.requirement[i][:2]:

                                    b.append(self.requirement[i])

                a += " ".join(b)

                if sub == 0:

                    return a

                else:

                    return b



                

            #Initialising Variables of requirement sets and subsets 

            self.requirementString2 = createSet(2,0)

            self.requirementSub2 = createSet(2,1)

            self.requirementString3 = createSet(3,0)

            self.requirementSub3 = createSet(3,1)

            self.requirementString4 = createSet(4,0)

            self.requirementSub4 = createSet(4,1)

            self.requirementString5 = createSet(5,0)

            self.requirementSub5 = createSet(5,1)

            self.requirementString6 = createSet(6,0)

            self.requirementSub6 = createSet(6,1)

            self.requirementString7 = createSet(7,0)

            self.requirementSub7 = createSet(7,1)

            self.requirementString8 = createSet(8,0)

            self.requirementSub8 = createSet(8,1)

                    

#--------------------------------------------JOB_DUTIES-------------------------------------------------------------                    

        with open (fpath, "r",encoding="latin-1") as file:

            self.dutiesList = []

            self.dutiesStr = ""

            for line in file:    

                if "DUTIES" in line:

                    self.dutiesList.append(list(islice(file, 2))[-1].strip())

                    #Debug exemptions with two blank lines instead of one

                    if self.dutiesList[0] == "" :

                        self.dutiesList[0] = list(islice(file, 1))[-1].strip()

                    while self.dutiesList[-1] != "" :

                        self.dutiesList.append(list(islice(file, 1))[-1].strip())

                        

                    #Delete last empty element of Array    

                    self.dutiesList = self.dutiesList[:-1]

                    

                    #Create string variable from List:

                    self.dutiesStr = self.dutiesList[0]

                    

#----------------------------------------------OPEN_DATE--------------------------------------------------------------

        with open (fpath, "r",encoding="latin-1") as file:            

            self.openDateStr = "11-11-1111"

            for line in file:                

                if "Open Date:" in line or "Open date:" in line and self.openDateStr == "11-11-1111":

                    self.openDateStr = line.split("ate:")[-1].strip()

                    if "(" in self.openDateStr:

                        self.openDateStr = self.openDateStr.split("(")[0].strip()

                        

            if len(self.openDateStr) <= 8:

                self.openDateObj = datetime.datetime.strptime(self.openDateStr, '%m-%d-%y')

            else:

                self.openDateObj = datetime.datetime.strptime(self.openDateStr, '%m-%d-%Y')

                    

            self.openDate = self.openDateObj.date()

                

#-----------------------------EXAM_TYPE, OPEN, INT_DEPT_PROM, DEPT_PROM, OPEN_INT_PROM--------------------------

        with open (fpath, "r",encoding="latin-1") as file:            

            self.examOpenTo = "####NO INFORMATION####"

            self.intDeptProm = 0

            self.openIntProm = 0

            self.deptProm = 0

            

            for line in file:

                

                if "(Exam Open to" in line and self.examOpenTo == "####NO INFORMATION####":

                    self.examOpenTo = line.split("pen to ")[1].strip().strip(")")

                    

                if "INTERDEPARTMENTAL" in line:

                    if "OPEN" in line or "OPEN" in list(islice(file, 1))[-1]:

                        self.openIntProm = 1

                    else:

                        self.intDeptProm = 1

                        

                if "A DEPARTMENTAL" in line:

                    self.deptProm = 1

            

            self.examType="####NO INFORMATION####"

            

            if self.openIntProm == 1:

                self.examType="OPEN_INT_PROM"

            if self.intDeptProm== 1:

                self.examType="INT_DEPT_PROM"

            if self.deptProm==1:

                self.examType="DEPT_PROM"

            if self.deptProm==0 and self.intDeptProm==0 and self.openIntProm==0 and "ll" in self.examOpenTo[1:3]:

                self.examType="OPEN"                    

#---------------------------------------DRIVERS_LICENSE_REQ-----------------------------------------

        with open (fpath, "r",encoding="latin-1") as file:            

            self.driveLic = "not required"

            for line in file:      

                if "license" in line or "License" in line:

                    #possibly required

                    if "may" in line and "require" in line and "valid California" in line and "river's license" in line:

                        self.driveLic = "possibly required"

                    if "California Driver's license may be required" in line:

                        self.driveLic = "possibly required"

                    if "For positions requiring a valid Class A or B driver's license" in line:

                        self.driveLic = "possibly required"

                    if "A valid Class A or B driver's license" in line and "may be required for some positions" in line:

                        self.driveLic = "possibly required"

                    if "Some positions may require a valid Class B driver's license" in line:

                        self.driveLic = "possibly required"

                    if "For positions requiring a valid Class A driver's license" in line:

                        self.driveLic = "possibly required"                     

                    if "river's license" in line and "is required" in line:

                        self.driveLic = "required"

                    if "valid California drivers' license is required" in line:

                        self.driveLic = "required"

                    if "Appointment is subject to possession of a valid California driver's license" in line:

                        self.driveLic = "required"

                    if "require an unrestricted California Class A driver's license" in line:

                        self.driveLic = "required"

                    if "A valid California driver's license and a good driving record are required" in line:

                        self.driveLic = "required"

                    if "A valid California Class B driver's license and valid medical certificate are required" in line:

                        self.driveLic = "required"

                    if "Possession of a valid California driver's license"in line:

                        self.driveLic = "required"

                    if "are required to maintain a valid Class B driver's license" in line:

                        self.driveLic = "required"

                    if "must attach a copy of their valid Class B California driver's license" in line:

                        self.driveLic = "required"

#------------------------------------------------DRIV_LIC_TYPE----------------------------------------------------------                        

        with open (fpath, "r",encoding="latin-1") as file:

            self.classA=0

            self.classB=0

            self.classC=0

            for line in file:

                if "Class" in line and "river" in line and "license" in line:

                    if "Class A" in line or "(or A)" in line:

                        self.classA = 1

                    if "Class B" in line or "(or B)" in line or " or B " in line or 'Class "B"' in line:

                        self.classB = 1

                    if "Class ''B''" in line:

                        self.classB = 1

                    if "Class C" in line or "(or C)" in line:

                        self.classC =1

                    if "B or C" in line:

                        self.classB = 1

                        self.classC = 1

                    if "1/A or 2/B" in line:

                        self.classA = 1

                        self.classB = 1



            licTypeList=[]

            

            if self.classA==1:

                licTypeList.append("A")

            if self.classB==1:

                licTypeList.append("B")

            if self.classC==1:

                licTypeList.append("C")

                

            self.licType=", ".join(licTypeList)

            

            if self.licType=="":

                self.licType="####NO INFORMATION####"



#-------------------ENTRY_SALARY_GEN,ENTRY_SALARY_DWP, ENTRY_SALARY_AIRPRT, ENTRY_SALARY_HRBR-------------------------------------------------------

        with open (fpath, "r",encoding="latin-1") as file:            

            self.salary_list = []

            self.salary_gen = ""

            self.salary_dwp = "####NO INFORMATION####"

            self.salary_airport = "####NO INFORMATION####"

            self.salary_harbor = "####NO INFORMATION####"

            

            for line in file:

                if "$" in line:

                    self.salary_list.append(line.strip())

                

            if len(self.salary_list) == 3 and self.salary_list[2][0]=="$":

                self.salary_list[1] = self.salary_list[2] + " " + self.salary_list[2]

                del self.salary_list[2]

            

            if len(self.salary_list) > 1 and "journey-level" in self.salary_list[1]:

                self.salary_list[0] += " / " + self.salary_list[1]

                    

            if len(self.salary_list) >= 1:

                if "Department of Water and Power" not in self.salary_list[0]:

                    self.salary_gen = self.salary_list[0].replace(",", "").replace(" to ","-").replace(" and ", ", ").replace("; ", ", ")

                    self.salary_gen = self.salary_gen.replace(";","").replace(".","")

                else:

                    self.salary_gen = "####NO INFORMATION####"

            else:

                self.salary_gen = "####NO INFORMATION####"

                

            if "effective" in self.salary_gen:

                self.salary_gen = self.salary_gen.split(" (")[0].replace("*","")

            

            self.salary_gen = self.salary_gen.replace("Fl","fl").replace("at ra","at-ra").replace(" flat-rated"," (flat-rated)")

                

            for element in self.salary_list:

                if "Department of Water and Power" in element:

                    self.salary_dwp = element.replace(",", "").replace(" to ","-").replace(" and ", ", ").replace("; ", ", ")

                    self.salary_dwp = self.salary_dwp.replace(";","").replace(".","")

                    self.salary_dwp = self.salary_dwp.replace("r, P","r and P")

                    position = self.salary_dwp.index('$')

                    self.salary_dwp = self.salary_dwp[(position):]

                    

                elif "Airport" in element:

                    self.salary_airport = element

                    position1 = self.salary_airport.index('$')

                    self.salary_airport = self.salary_airport[(position1):]

                    self.salary_airport = self.salary_airport.replace(" to ","-").replace(",","").replace(".","")

                    

                elif "Harbor" in element:

                    self.salary_harbor = element

                    position2 = self.salary_harbor.index('$')

                    self.salary_harbor = self.salary_harbor[(position2):]

                    self.salary_harbor = self.salary_harbor.replace(" to ","-").replace(",","").replace(".","")

                    

            if "Harbor" in self.salary_dwp:

                self.salary_harbor = self.salary_dwp.split("Harbor Department is ")[1]

                self.salary_dwp = self.salary_dwp.split(" ")[0]

            if " The salary range in the Department of Water and Power is " in self.salary_dwp:

                self.salary_gen = self.salary_dwp.split(" The salary range in the Department of Water and Power is ")[0]

                self.salary_dwp = self.salary_dwp.split(" The salary range in the Department of Water and Power is ")[1]

            

            self.salary_dwp = self.salary_dwp.replace("Fl","fl").replace("at ra","at-ra").replace(" flat-rated"," (flat-rated)")

 #------------------------------------------------------ADDTL_LIC--------------------------------------------------------               

        licensesList=['Registration as a licensed Land Surveyor by the California State Board of Registration for Professional Engineers and Land Surveyor',

                      'License to Act as a Journey level Elevator Mechanic issued by the City of Los Angeles Department of Building and Safetylicense to operate a forklift or crane',

                      'Physician Assistant with documentation of education and training as a Physician Assistant in family practice, adult, or emergency medicine',

                      'United States Coast Guard Master and/or Ship Pilot License', 'Qualified Applicator for pest control activities',

                      'Drug Enforcement Administration (DEA) registration number', 'T2 (or higher) Water Treatment Operator license and a D1 Water Distribution Operator License',

                      'certificate as a Registered Deputy Building Inspector in one or more of the specialized certifications issued by the Department of Building and Safety of the City of Los Angeles',

                      'State of California Paramedic License issued by the Emergency Medical Services Authority', 'California License as a Licensed Vocational Nurse',

                      "Radiograph or Radiotelephone General Class Operator's License issued by the Federal Communications Commission",

                      'licensed general contractor', 'Professional Engineer License', 'license as a certified welder for structural steel issued by the Los Angeles City Department of Building and Safety',

                      'Registered Nurse license', 'Agricultural Pest Control Advisor with certification for control of weeds, plant pathogens, or insects, mites, and other invertebrates',

                      'Registration as a Professional Engineer or possession of both Registered Geologist and Engineering Geologist licenses issued by the California Board for Professional Engineers, Land Surveyors, and Geologists',

                      'license as a Registered Deputy Building Inspector in one or more of the specialized certifications issued by the Department of Building and Safety of the City of Los Angeles',

                      "first-class pilot's license", "General Radiotelephone Operator's license issued by the Federal Communications Commission",

                      'Professional Geologist and Certified Engineering Geologist licenses', 'licensed contractor in the carpentry, masonry, electrical, plumbing, or heating construction trades',

                      'State of California Certificate Appraisal License', 'Registration as a Land Surveyor with the State of California Board for Professional Engineers, Land Surveyors and Geologists',

                      'United States Coast Guard certification  as a First Class Pilot for the San Pedro Bay and its approaches and tributaries with unlimited tonnage',

                      'Certified Commercial Applicators license, issued by the State of California, Department of Food and Agricultur',

                      'Registration as a Professional Engineer with the California Board for Professional Engineers, Land Surveyors, and Geologists',

                      'Registered Nurse', 'Paramedic License issued by the State of California Emergency Medical Services Authority with paramedic accreditation by the County of Los Angeles Department of Health Services',

                      "journeyman plumber's license issued by the City of Los Angeles Department of Building and Safety",

                      'mechanical license with Airframe and Power Plant ratings issued by the Federal Aviation Administration',

                      'T3 (or higher) Water Treatment Operator license', 'license as a Licensed Clinical Social Worker',

                      "Radiotelegraph or Radiotelephone General Class Operator's License issued by the Federal Communications Commission",

                      'Furnishing Number', 'Distribution Operator License D-4 or D-5', 'Architect license', "Pest Control Qualified Applicator's License or Qualified Applicator's Certificate or a Pest Control Advisor's License",

                      'Selective certifications 1A, 1C, 1D, 1E, and 1F require a Certificate of Completion or valid license', 'T4 (or higher) Water Treatment Operator license and a valid D1 (or higher) Water Distribution Operator license issued by the California State Department of Water Resources, Division of Drinking Water',

                      'Registration as a Civil Engineer with the California State Board of Registration for Professional Engineers',

                      'United States Coast Guard licensed Master, Staff Captain, or Chief Mate',

                      'registered as a Professional Engineer', 'Structural Engineer License',

                      'Paramedic license issued by the State of California with accreditation by the Local Emergency Services Agency of Los Angeles County',

                      'Professional Engineer license', 'MICN certification', 'license to operate a forklift or crane',

                      'Professional Engineer (PE) license', 'Registered Nurse License', 'Crane Operator certification for Small and Large Telescopic Mobile Cranes license issued by the NCCCO',

                      "Qualified Applicator's License", 'journey-level plumber license', 'Grade 5 State of California Treatment Operator License',

                      'Registration as a Professional Engineer in Traffic Engineering or Civil Engineering with the California State Board for Professional Engineers, Land Surveyors, and Geologists',

                      "California Department of Pesticide Regulation, Pest Control Advisor License and/or a Qualified Applicator's Certificate",

                      'Professional Land Surveyor or Professional Engineer license', 'California Architect license issued by the California Architects Board',

                      'California Bureau of Automotive Repair Lamp Adjuster License', 'Paramedic license', 'Competent Conveyance (Elevator) Mechanic certification issued by the State of California',

                      'Professional Engineer (PE) registration', 'registration as a Professional Geologist with the California State Board for Professional Engineers, Land Surveyors, and Geologists',

                      'Qualified Test and Repair Mechanic License', 'license as a certified Welder for Structural Steel issued by the City of Los Angeles, Department of Building and Safety',

                      'professional engineer license', 'Brake Adjuster License', 'Civil Engineering License', 'Engineer-In-Training (EIT) certification',

                      'Registration as a Professional Engineer with the State of California Board for Professional Engineers, Land Surveyors, and Geologists with a degree in Chemical or Mechanical Engineering',

                      "Crane/Derrick Surveyor license issued by the State of California's Division of Occupational Safety and Health",

                      'Engineer-in-Training (EIT) certification', 'Water Distribution Operator License', 'City of Los Angeles Journey-level Plumbers license',

                      "unlimited Steam Engineer's license issued by the Department of Building and Safety of the City of Los Angeles",

                      'certified as a Nurse Practitioner by the California Board of Registered Nursing', 'Grade D-3 State of California Water Resources Control Board Water Distribution Operator License',

                      'California Department of Public Health D1 or D2 Water Distribution Operator License', 'Occupational Therapist license',

                      "Rubber Tired Tractor B Operator's license and a Hydrocrane Operator's license", 'FCC General Radiotelephone', 'Professional Engineer in Structural Engineering license',

                      "certification as a Leadership in Energy and Environmental Design (LEED) Green Associate by the United States Green Building Council"]

        

        self.otherLic=[]

        with open (fpath, "r",encoding="latin-1") as file:

            for line in file:

                for element in licensesList:

                    if element in line and element not in self.otherLic:

                        self.otherLic.append(element)

                        



        jList=[]

        for i in range(0,len(self.otherLic)):

            for j in range(0,len(self.otherLic)):

                if self.otherLic[j] in self.otherLic[i] and not self.otherLic[j] == self.otherLic[i]:

                    jList.append(j)

                if self.otherLic[j] == self.otherLic[i] and j!=i:

                    jList.append(j)

        for j in jList:

            #print(self.jobTitle)

            #print(self.otherLic)

            del self.otherLic[j]



                    

#--------------------------------------------------FULL_TIME_PART_TIME---------------------------------------------------

        def fullpart(setNo):

            fullpart=""

            if "ull-time" in setNo and "art-time" in setNo:

                fullpart="full-/part-time"

            if "ull-time" in setNo and "art-time" not in setNo:

                fullpart="full-time"

            if "ull-time" not in setNo and "art-time" in setNo:

                fullpart="part-time"



            return fullpart

        

        self.fullpart1=fullpart(self.requirementString1)

        self.fullpart2=fullpart(self.requirementString2)

        self.fullpart3=fullpart(self.requirementString3)

        self.fullpart4=fullpart(self.requirementString4)

        self.fullpart5=fullpart(self.requirementString5)

        self.fullpart6=fullpart(self.requirementString6)

        self.fullpart7=fullpart(self.requirementString7)

        self.fullpart8=fullpart(self.requirementString8)



#--------------------------------------------EXPERIENCE_LENGTH--------------------------------------------



        def expLen(setNo):

            expYearList=[]

            if "-time" in setNo or "hours of experience" in setNo:

                wordList= setNo.split(" ")

                for i in range(0,len(wordList)):

                    if "-time" in wordList[i]:   

                        if "hours" ==wordList[i-2]:

                            expYear=wordList[i-4]+" hours"

                            expYearList.append(expYear)

                        elif "months" ==wordList[i-2]:

                            expYear=wordList[i-3]+" months"

                            expYearList.append(expYear)

                        elif "of" ==wordList[i-1] and "-year" in wordList[i-2]:

                            expYear=wordList[i-2].split("-")[0]

                            expYearList.append(expYear)

                        elif "of" ==wordList[i-1] and "-year" not in wordList[i-2]:

                            expYear = wordList[i-3]

                            if "-half" in wordList[i-3]:

                                expYear = wordList[i-5] + " " + wordList[i-4] + " " + wordList[i-3]

                            expYearList.append(expYear)   

                        elif "ear" in wordList[i-1]:

                            expYear = wordList[i-2]

                            expYearList.append(expYear)

                        elif "-year" in wordList[i-1]:

                            expYear=wordList[i-1].split("-")[0]

                            expYearList.append(expYear)

                    if "hours" in wordList[i] and "experience" in wordList[i+2]:

                        if "cumulative" in wordList[i-1]:

                            expYear=wordList[i-2]+" hours"

                            expYearList.append(expYear)

                        else:

                            expYear=wordList[i-1]+" hours"

                            expYearList.append(expYear)

                    if "hours" in wordList[i] and "City" in wordList[i+3]:

                        expYear=wordList[i-1]+" hours"

                        expYearList.append(expYear)

                    if "hours" in wordList[i] and "paid" in wordList[i+2] and "experience" in wordList[i+3]:

                        expYear=wordList[i-1]+" hours"

                        expYearList.append(expYear)

                        

            for j in range(0, len(expYearList)):

                if "(" in expYearList[j]:

                    expYearList[j]=expYearList[j].replace("(","").replace(")","")

                if "second-level" in expYearList[j]:

                    expYearList[j]=""

                if "months" in expYearList[j]:

                    if "six " in expYearList[j] or "Six " in expYearList[j] or "6 " in expYearList[j]:

                        expYearList[j]="1/2"

                    if "Eight " in expYearList[j] or "eight " in expYearList[j] or "8 " in expYearList[j]:

                        expYearList[j]="2/3"

                    if "Eighteen" in expYearList[j] or "eighteen" in expYearList[j]:

                        expYearList[j]="1.5"

                    if "Three" in expYearList[j] or "three" in expYearList[j] or "3 " in expYearList[j]:

                        expYearList[j]="1/4"

                if expYearList[j]=="Two and one-half" or expYearList[j]=="two and one-half":

                    expYearList[j]="2.5"

                if expYearList[j]=="one" or expYearList[j]=="One":

                    expYearList[j]="1"

                if expYearList[j]=="two" or expYearList[j]=="Two":

                    expYearList[j]="2"

                if expYearList[j]=="three" or expYearList[j]=="Three":

                    expYearList[j]="3"

                if expYearList[j]=="four" or expYearList[j]=="Four":

                    expYearList[j]="4"

                if expYearList[j]=="five" or expYearList[j]=="Five":

                    expYearList[j]="5"

                if expYearList[j]=="six" or expYearList[j]=="Six":

                    expYearList[j]="6"

                if expYearList[j]=="seven" or expYearList[j]=="Seven":

                    expYearList[j]="7"

                if expYearList[j]=="eight" or expYearList[j]=="Eight":

                    expYearList[j]="8"

                if expYearList[j]=="Nine" or expYearList[j]=="nine":

                    expYearList[j]="9"

                if expYearList[j]=="twelve" or expYearList[j]=="Twelve":

                    expYearList[j]="12"

                if "(" in expYearList[j]:

                    expYearList[j]=expYearList[j].replace("(","").replace(")","")

            return " | ".join(expYearList).strip().strip(" | ")

        

        self.expYearList1=expLen(self.requirementString1)

        self.expYearList2=expLen(self.requirementString2)

        self.expYearList3=expLen(self.requirementString3)

        self.expYearList4=expLen(self.requirementString4)

        self.expYearList5=expLen(self.requirementString5)

        self.expYearList6=expLen(self.requirementString6)

        self.expYearList7=expLen(self.requirementString7)

        self.expYearList8=expLen(self.requirementString8)





#--------------------------------------------EXP_JOB_CLASS_TITLE/_ALT_RESP----------------------------------

        jobTitles=['311 DIRECTOR', 'ACCOUNTANT', 'ACCOUNTING CLERK', 'ACCOUNTING RECORDS SUPERVISOR','ADMINISTRATIVE ANALYST'

           , 'ADMINISTRATIVE CLERK', 'ADMINISTRATIVE HEARING EXAMINER', 'ADVANCE PRACTICE PROVIDER CORRECTIONAL CARE'

           , 'AIR CONDITIONING MECHANIC', 'AIR CONDITIONING MECHANIC SUPERVISOR', 'AIRPORT AIDE'

           , 'AIRPORT CHIEF INFORMATION SECURITY OFFICER', 'AIRPORT ENGINEER', 'AIRPORT GUIDE'

           , 'AIRPORT INFORMATION SPECIALIST', 'AIRPORT LABOR RELATIONS ADVOCATE', 'AIRPORT MANAGER'

           , 'AIRPORT POLICE CAPTAIN', 'AIRPORT POLICE LIEUTENANT', 'AIRPORT POLICE OFFICER'

           , 'AIRPORT POLICE SPECIALIST', 'AIRPORT SUPERINTENDENT OF OPERATIONS'

           , 'AIRPORTS MAINTENANCE SUPERINTENDENT', 'AIRPORTS MAINTENANCE SUPERVISOR'

           , 'AIRPORTS PUBLIC AND COMMUNITY RELATIONS DIRECTOR', 'ANIMAL CARE ASSISTANT'

           , 'ANIMAL CARE TECHNICIAN', 'WATER TREATMENT OPERATOR', 'ANIMAL CONTROL OFFICER'

           , 'ANIMAL KEEPER', 'APPARATUS OPERATOR', 'APPLICATIONS PROGRAMMER', 'APPRENTICE - METAL TRADES'

           , 'APPRENTICE MACHINIST', 'AQUARIST', 'AQUARIUM EDUCATOR', 'AQUATIC DIRECTOR', 'AQUATIC FACILITY MANAGER'

           , 'AQUEDUCT AND RESERVOIR KEEPER', 'AQUEDUCT AND RESERVOIR SUPERVISOR', 'ARCHITECT'

           , 'CAMPUS INTERVIEWS ONLY', 'ARCHITECTURAL DRAFTING TECHNICIAN', 'ARCHIVIST', 'ART CENTER DIRECTOR'

           , 'ART CURATOR', 'ART INSTRUCTOR', 'ARTS ASSOCIATE', 'ARTS MANAGER', 'ASBESTOS SUPERVISOR'

           , 'ASBESTOS WORKER', 'ASPHALT PLANT OPERATOR', 'ASPHALT PLANT SUPERVISOR', 'ASSISTANT AIRPORT MANAGER'

           , 'ASSISTANT COMMUNICATIONS CABLE WORKER', 'ASSISTANT COMMUNICATIONS ELECTRICIAN'

           , 'ASSISTANT DEPUTY SUPERINTENDENT OF BUILDING', 'ASSISTANT DIRECTOR INFORMATION SYSTEMS'

           , 'ASSISTANT GARDENER', 'ASSISTANT INSPECTOR', 'ASSISTANT RETIREMENT PLAN MANAGER'

           , 'ASSISTANT SIGNAL SYSTEMS ELECTRICIAN', 'ASSISTANT STREET LIGHTING ELECTRICIAN'

           , 'ASSISTANT TREE SURGEON', 'ASSISTANT UTILITY BUYER', 'ASSOCIATE ZONING ADMINISTRATOR'

           , 'AUDIO VISUAL TECHNICIAN', 'AUDITOR', 'AUTO BODY BUILDER AND REPAIRER', 'AUTO BODY REPAIR SUPERVISOR'

           , 'AUTO ELECTRICIAN', 'AUTO PAINTER', 'AUTOMOTIVE DISPATCHER', 'AUTOMOTIVE SUPERVISOR'

           , 'AVIONICS SPECIALIST', 'BACKGROUND INVESTIGATION MANAGER', 'BACKGROUND INVESTIGATOR'

           , 'BENEFITS SPECIALIST', 'BLACKSMITH', 'BOILERMAKER', 'BOILERMAKER SUPERVISOR', 'BUILDING CIVIL ENGINEER'

           , 'BUILDING CONSTRUCTION AND MAINTENANCE', 'BUILDING ELECTRICAL ENGINEER', 'BUILDING INSPECTOR'

           , 'BUILDING MAINTENANCE DISTRICT SUPERVISOR', 'BUILDING MECHANICAL ENGINEER'

           , 'BUILDING MECHANICAL INSPECTOR', 'BUILDING OPERATING ENGINEER', 'BUILDING REPAIR SUPERVISOR'

           , 'BUILDING REPAIRER', 'BUS OPERATOR', 'BUS OPERATOR SUPERVISOR', 'CABLE TELEVISION PRODUCTION MANAGER'

           , 'CARPENTER', 'CARPENTER SUPERVISOR', 'CARPET LAYER', 'CEMENT FINISHER', 'CEMENT FINISHER SUPERVISOR'

           , 'CEMENT FINISHER WORKER', 'CHEMIST', 'CHIEF ADMINISTRATIVE ANALYST', 'CHIEF AIRPORTS ENGINEER'

           , 'CHIEF BENEFITS ANALYST', 'CHIEF BUILDING OPERATING ENGINEER', 'CHIEF CLERK', 'CHIEF CLERK PERSONNEL'

           , 'CHIEF CLERK POLICE', 'CHIEF CLERK POLICE', 'CHIEF COMMUNICATIONS OPERATOR', 'CHIEF CONSTRUCTION INSPECTOR', 'CHIEF CUSTODIAN SUPERVISOR', 'CHIEF ELECTRIC PLANT OPERATOR', 'CHIEF ENVIRONMENTAL COMPLIANCE INSPECTOR', 'CHIEF FINANCIAL OFFICER', 'CHIEF FORENSIC CHEMIST', 'CHIEF HARBOR ENGINEER', 'CHIEF INSPECTOR', 'CHIEF INTERNAL AUDITOR', 'CHIEF MANAGEMENT ANALYST', 'CHIEF OF AIRPORT PLANNING', 'CHIEF OF DRAFTING OPERATIONS', 'CHIEF OF OPERATIONS', 'CHIEF OF PARKING ENFORCEMENT OPERATIONS', 'CHIEF PARK RANGER', 'CHIEF PORT PILOT', 'CHIEF SAFETY ENGINEER PRESSURE VESSELS', 'CHIEF SECURITY OFFICER', 'CHIEF STREET SERVICES INVESTIGATOR', 'CHIEF TAX COMPLIANCE OFFICER', 'CHIEF TRANSPORTATION INVESTIGATOR', 'CITY PLANNER', 'CITY PLANNING ASSOCIATE', 'CIVIL ENGINEER', 'CIVIL ENGINEERING ASSOCIATE', 'CIVIL ENGINEERING DRAFTING TECHNICIAN', 'CLAIMS AGENT', 'COMMERCIAL FIELD REPRESENTATIVE', 'COMMERCIAL FIELD SUPERVISOR', 'COMMERCIAL SERVICE SUPERVISOR', 'COMMISSION EXECUTIVE ASSISTANT', 'COMMUNICATIONS CABLE SUPERVISOR', 'COMMUNICATIONS CABLE WORKER', 'COMMUNICATIONS ELECTRICIAN', 'COMMUNICATIONS ELECTRICIAN SUPERVISOR', 'COMMUNICATIONS ENGINEER', 'COMMUNICATIONS ENGINEERING ASSOCIATE', 'COMMUNICATIONS INFORMATION REPRESENTATIVE', 'COMMUNITY AFFAIRS ADVOCATE', 'COMMUNITY HOUSING PROGRAMS MANAGER', 'COMMUNITY PROGRAM ASSISTANT', 'COMPLIANCE PROGRAM MANAGER', 'CONSTRUCTION AND MAINTENANCE SUPERINTENDENT', 'CONSTRUCTION AND MAINTENANCE SUPERVISOR', 'CONSTRUCTION EQUIPMENT SERVICE WORKER', 'CONSTRUCTION ESTIMATOR', 'CONSTRUCTION INSPECTOR', 'CONTRACT ADMINISTRATOR', 'CONTROL SYSTEMS ENGINEERING ASSOCIATE', 'CORRECTIONAL NURSE', 'CRIME AND INTELLIGENCE ANALYST', 'CRIMINALIST', 'CUSTODIAL SERVICES ASSISTANT', 'CUSTODIAN', 'CUSTODIAN SUPERVISOR', 'CUSTOMER SERVICE REPRESENTATIVE', 'CUSTOMER SERVICE SPECIALIST', 'DATA PROCESSING TECHNICIAN', 'DATABASE ARCHITECT', 'DECK HAND', 'DELIVERY DRIVER', 'DEPARTMENTAL CHIEF ACCOUNTANT', 'DETENTION OFFICER', 'DIRECTOR OF AIRPORT MARKETING', 'DIRECTOR OF AIRPORT OPERATIONS', 'DIRECTOR OF AIRPORTS ADMINISTRATION\tREVISED', 'DIRECTOR OF COMMUNICATIONS SERVICES', 'DIRECTOR OF ENFORCEMENT OPERATIONS', 'DIRECTOR OF FIELD OPERATIONS', 'DIRECTOR OF HOUSING', 'DIRECTOR OF MAINTENANCE AIRPORTS', 'DIRECTOR OF POLICE TRANSPORTATION', 'DIRECTOR OF PORT CONSTRUCTION AND MAINTENANCE', 'DIRECTOR OF PRINTING SERVICES', 'DIRECTOR OF SECURITY SERVICES', 'DIRECTOR OF SYSTEMS', 'DISTRICT SUPERVISOR ANIMAL SERVICES', 'DIVISION LIBRARIAN', 'DRILL RIG OPERATOR', 'DUPLICATING MACHINE OPERATOR', 'ELECTRIC DISTRIBUTION MECHANIC', 'ELECTRIC DISTRIBUTION MECHANIC SUPERVISOR', 'ELECTRIC METER SETTER', 'ELECTRIC SERVICE REPRESENTATIVE', 'ELECTRIC STATION OPERATOR', 'ELECTRIC TROUBLE DISPATCHER', 'ELECTRICAL CRAFT HELPER', 'ELECTRICAL ENGINEERING ASSOCIATE', 'ELECTRICAL ENGINEERING DRAFTING TECHNICIAN', 'ELECTRICAL INSPECTOR', 'ELECTRICAL MECHANIC', 'ELECTRICAL MECHANIC SUPERVISOR', 'ELECTRICAL REPAIR SUPERVISOR', 'ELECTRICAL REPAIRER', 'ELECTRICAL SERVICE WORKER', 'ELECTRICAL SERVICES MANAGER', 'ELECTRICAL TESTER', 'ELECTRICIAN', 'ELECTRICIAN SUPERVISOR', 'ELEVATOR MECHANIC', 'ELEVATOR MECHANIC HELPER', 'ELEVATOR REPAIR SUPERVISOR', 'EMERGENCY MANAGEMENT COORDINATOR', 'EMERGENCY MEDICAL SERVICES (EMS) EDUCATOR', 'EMS ADVANCED PROVIDER', 'EMS NURSE PRACTITIONER SUPERVISOR', 'ENGINEER OF FIRE DEPARTMENT', 'ENGINEER OF SURVEYS', 'ENGINEERING DESIGNER', 'ENGINEERING GEOLOGIST', 'ENGINEERING GEOLOGIST ASSOCIATE', 'ENVIRONMENTAL AFFAIRS OFFICER', 'ENVIRONMENTAL COMPLIANCE INSPECTOR', 'ENVIRONMENTAL ENGINEER', 'CAMPUS INTERVIEWS ONLY', 'ENVIRONMENTAL SPECIALIST', 'ENVIRONMENTAL SUPERVISOR', 'EQUIPMENT MECHANIC (Automotive Mechanic)', 'EQUIPMENT OPERATOR', 'EQUIPMENT REPAIR SUPERVISOR', 'EQUIPMENT SPECIALIST', 'EQUIPMENT SUPERINTENDENT', 'EQUIPMENT SUPERVISOR', 'EXAMINER OF QUESTIONED DOCUMENTS', 'EXECUTIVE ADMINISTRATIVE ASSISTANT', 'EXECUTIVE ASSISTANT AIRPORTS', 'EXHIBIT PREPARATOR', 'FIELD ENGINEERING AIDE', 'FINANCIAL ANALYST', 'FINANCIAL DEVELOPMENT OFFICER', 'FINANCIAL MANAGER', 'FINGERPRINT IDENTIFICATION EXPERT', 'FIRE ASSISTANT CHIEF', 'FIRE BATTALION CHIEF', 'FIRE CAPTAIN', 'FIRE HELICOPTER PILOT', 'FIRE INSPECTOR', 'FIRE PROTECTION ENGINEERING ASSOCIATE', 'FIRE SPECIAL INVESTIGATOR', 'FIRE SPRINKLER INSPECTOR', 'FIREARMS EXAMINER', 'FIREBOAT MATE', 'FIREBOAT PILOT', 'FIREFIGHTER', 'FISCAL SYSTEMS SPECIALIST', 'FLEET SERVICES MANAGER', 'FORENSIC PRINT SPECIALIST', 'GALLERY ATTENDANT', 'GARAGE ASSISTANT', 'GARAGE ATTENDANT', 'GARDENER CARETAKER', 'GENERAL AUTOMOTIVE SUPERVISOR', 'GENERAL SERVICES MANAGER', 'GEOGRAPHIC INFORMATION SYSTEMS CHIEF', 'GEOGRAPHIC INFORMATION SYSTEMS SPECIALIST', 'GEOGRAPHIC INFORMATION SYSTEMS SUPERVISOR', 'GEOTECHNICAL ENGINEER', 'GOLF STARTER', 'GOLF STARTER SUPERVISOR', 'GRAPHICS DESIGNER', 'GRAPHICS SUPERVISOR', 'HARBOR ENGINEER', 'HARBOR PLANNING AND ECONOMIC ANALYST', 'HARBOR PLANNING AND RESEARCH DIRECTOR', 'HEAD CUSTODIAN SUPERVISOR', 'HEATING AND REFRIGERATION INSPECTOR', 'HEAVY DUTY EQUIPMENT MECHANIC', 'HEAVY DUTY TRUCK OPERATOR', 'HELICOPTER MECHANIC', 'HELICOPTER MECHANIC SUPERVISOR', 'HOUSING INSPECTOR', 'HOUSING INVESTIGATOR', 'HOUSING PLANNING AND ECONOMIC ANALYST', 'HUMAN RELATIONS ADVOCATE', 'HYDROGRAPHER', 'IMPROVEMENT ASSESSOR SUPERVISOR', 'INDUSTRIAL AND COMMERCIAL FINANCE OFFICER', 'INDUSTRIAL CHEMIST', 'INDUSTRIAL GRAPHICS SUPERVISOR', 'INDUSTRIAL HYGIENIST', 'INFORMATION SERVICES SPECIALIST', 'INFORMATION SYSTEMS MANAGER', 'INSTRUMENT MECHANIC', 'INSTRUMENT MECHANIC SUPERVISOR', 'INTERNAL AUDITOR', 'INVESTMENT OFFICER', 'IRRIGATION SPECIALIST', 'LABOR SUPERVISOR', 'LABORATORY TECHNICIAN', 'LAND SURVEYING ASSISTANT', 'LANDSCAPE ARCHITECT', 'LANDSCAPE ARCHITECTURAL ASSOCIATE', 'LEGISLATIVE ASSISTANT', 'LEGISLATIVE REPRESENTATIVE', 'LIBRARIAN', 'LIBRARY ASSISTANT', 'LICENSED VOCATIONAL NURSE', 'LINE MAINTENANCE ASSISTANT', 'LOCKSMITH', 'MACHINIST', 'MACHINIST SUPERVISOR', 'MAINTENANCE AND CONSTRUCTION HELPER', 'MAINTENANCE LABORER', 'MANAGEMENT AIDE', 'MANAGEMENT ANALYST', 'MANAGEMENT ASSISTANT', 'MANAGING WATER UTILITY ENGINEER', 'MARINE AQUARIUM CURATOR', 'MARINE AQUARIUM PROGRAM DIRECTOR', 'MARINE ENVIRONMENTAL MANAGER', 'MARINE ENVIRONMENTAL SUPERVISOR', 'MASONRY WORKER', 'MATERIALS TESTING ENGINEERING ASSOCIATE', 'MATERIALS TESTING TECHNICIAN', 'MECHANICAL ENGINEER', 'MECHANICAL ENGINEERING ASSOCIATE', 'MECHANICAL ENGINEERING DRAFTING TECHNICIAN', 'MECHANICAL HELPER', 'MECHANICAL REPAIR GENERAL SUPERVISOR', 'MECHANICAL REPAIR SUPERVISOR', 'MECHANICAL REPAIRER', 'MEDICAL ASSISTANT', 'METER READER', 'MOTION PICTURE AND TELEVISION MANAGER', 'MOTOR SWEEPER OPERATOR', 'OCCUPATIONAL HEALTH NURSE', 'OFFICE ENGINEERING TECHNICIAN', 'OFFICE SERVICES ASSISTANT', 'OFFICE TRAINEE', 'OPERATIONS AND STATISTICAL RESEARCH ANALYST', 'PAINTER', 'PAINTER SUPERVISOR', 'PARK MAINTENANCE SUPERVISOR', 'PARK RANGER', 'PARK SERVICES ATTENDANT', 'PARK SERVICES SUPERVISOR', 'PARKING ATTENDANT', 'PARKING ENFORCEMENT MANAGER', 'PARKING MANAGER', 'PARKING METER TECHNICIAN', 'PARKING METER TECHNICIAN SUPERVISOR', 'PAYROLL ANALYST', 'PAYROLL SUPERVISOR', 'PERFORMING ARTS DIRECTOR', 'PERSONNEL ANALYST', 'PERSONNEL DIRECTOR', 'PERSONNEL RECORDS SUPERVISOR', 'PERSONNEL RESEARCH ANALYST', 'PHOTOGRAPHER', 'PILE DRIVER WORKER', 'PIPEFITTER', 'PIPEFITTER SUPERVISOR', 'PLANNING ASSISTANT', 'PLUMBER', 'PLUMBER SUPERVISOR', 'PLUMBING INSPECTOR', 'POLICE ADMINISTRATOR', 'POLICE CAPTAIN', 'POLICE COMMANDER', 'POLICE DETECTIVE', 'POLICE LIEUTENANT', 'POLICE OFFICER', 'POLICE PERFORMANCE AUDITOR', 'POLICE SERGEANT', 'POLICE SERVICE REPRESENTATIVE', 'POLICE SPECIAL INVESTIGATOR', 'POLICE SPECIALIST', 'POLICE SURVEILLANCE SPECIALIST', 'POLYGRAPH EXAMINER', 'PORT ELECTRICAL MECHANIC', 'PORT ELECTRICAL MECHANIC SUPERVISOR', 'PORT MAINTENANCE SUPERVISOR', 'PORT PILOT', 'PORT POLICE CAPTAIN', 'PORT POLICE LIEUTENANT', 'PORT POLICE OFFICER', 'PORT POLICE SERGEANT', 'PORTFOLIO MANAGER', 'POWER ENGINEERING MANAGER', 'POWER SHOVEL OPERATOR', 'PRE-PRESS OPERATOR', 'PRINCIPAL ACCOUNTANT', 'PRINCIPAL ANIMAL KEEPER', 'PRINCIPAL CITY PLANNER', 'PRINCIPAL CIVIL ENGINEER', 'PRINCIPAL CIVIL ENGINEERING DRAFTING TECHNICIAN', 'PRINCIPAL CLERK', 'PRINCIPAL CLERK POLICE', 'PRINCIPAL CLERK UTILITY', 'PRINCIPAL COMMUNICATIONS OPERATOR', 'PRINCIPAL CONSTRUCTION INSPECTOR', 'PRINCIPAL DEPUTY CONTROLLER', 'PRINCIPAL DETENTION OFFICER', 'PRINCIPAL ELECTRIC TROUBLE DISPATCHER', 'PRINCIPAL ELECTRICAL ENGINEERING DRAFTING TECHNICIAN', 'PRINCIPAL ENVIRONMENTAL ENGINEER', 'PRINCIPAL GROUNDS MAINTENANCE SUPERVISOR', 'PRINCIPAL INSPECTOR', 'PRINCIPAL LIBRARIAN', 'PRINCIPAL MECHANICAL ENGINEERING DRAFTING TECHNICIAN', 'PRINCIPAL PHOTOGRAPHER', 'PRINCIPAL PROPERTY OFFICER', 'PRINCIPAL PUBLIC RELATIONS REPRESENTATIVE', 'PRINCIPAL RECREATION SUPERVISOR', 'PRINCIPAL SECURITY OFFICER', 'PRINCIPAL STOREKEEPER', 'PRINCIPAL TAX AUDITOR', 'PRINCIPAL TAX COMPLIANCE OFFICER', 'PRINCIPAL TRANSPORTATION ENGINEER', 'PRINCIPAL UTILITY ACCOUNTANT', "PRINCIPAL WORKERS' COMPENSATION ANALYST", 'PRINTING PRESS OPERATOR', 'PRINTING SERVICES SUPERINTENDENT', 'PROCUREMENT ANALYST', 'PROCUREMENT SUPERVISOR', 'PROGRAMMER ANALYST', 'PROPERTY MANAGER', 'PROPERTY OFFICER', 'PROTECTIVE COATING WORKER', 'PUBLIC INFORMATION DIRECTOR', 'PUBLIC RELATIONS SPECIALIST', 'PUBLIC SAFETY RISK MANAGER', 'RATES MANAGER', 'REAL ESTATE ASSOCIATE', 'REAL ESTATE OFFICER', 'REAL ESTATE TRAINEE', 'RECREATION COORDINATOR', 'RECREATION FACILITY DIRECTOR', 'RECREATION SUPERVISOR', 'REFUSE COLLECTION SUPERVISOR', 'REFUSE COLLECTION TRUCK OPERATOR', 'REFUSE CREW FIELD INSTRUCTOR', 'REHABILITATION CONSTRUCTION SPECIALIST', 'REHABILITATION PROJECT COORDINATOR', 'REINFORCING STEEL WORKER', 'REPROGRAPHICS OPERATOR', 'REPROGRAPHICS SUPERVISOR', 'RETIREMENT PLAN MANAGER', 'RIDESHARE PROGRAM ADMINISTRATOR', 'RISK AND INSURANCE ASSISTANT', 'RISK MANAGEMENT AND PREVENTION PROGRAM SPECIALIST', 'RISK MANAGER', 'ROOFER', 'ROOFER SUPERVISOR', 'SAFETY ADMINISTRATOR', 'SAFETY ENGINEER', 'SAFETY ENGINEER ELEVATORS', 'SAFETY ENGINEER PRESSURE VESSELS', 'SAFETY ENGINEERING ASSOCIATE', 'SANITATION SOLID RESOURCES MANAGER', 'SANITATION WASTEWATER MANAGER', 'SECRETARY', 'SECRETARY LEGAL', 'SECURITY AIDE', 'SECURITY OFFICER', 'SENIOR ACCOUNTANT', 'SENIOR ADMINISTRATIVE ANALYST', 'SENIOR ADMINISTRATIVE CLERK', 'SENIOR ANIMAL CONTROL OFFICER', 'SENIOR ANIMAL KEEPER', 'SENIOR ARCHITECT', 'SENIOR ARCHITECTURAL DRAFTING TECHNICIAN', 'SENIOR AUDITOR', 'SENIOR AUTOMOTIVE SUPERVISOR', 'SENIOR BUILDING INSPECTOR', 'SENIOR BUILDING MECHANICAL INSPECTOR', 'SENIOR BUILDING OPERATING ENGINEER', 'SENIOR CARPENTER', 'SENIOR CHEMIST', 'SENIOR CITY PLANNER', 'SENIOR CIVIL ENGINEER', 'SENIOR CIVIL ENGINEERING DRAFTING TECHNICIAN', 'SENIOR CLAIMS REPRESENTATIVE', 'SENIOR COMMERCIAL FIELD REPRESENTATIVE', 'SENIOR COMMUNICATIONS CABLE WORKER', 'SENIOR COMMUNICATIONS ELECTRICIAN', 'SENIOR COMMUNICATIONS ELECTRICIAN SUPERVISOR', 'SENIOR COMMUNICATIONS ENGINEER', 'SENIOR COMMUNICATIONS OPERATOR', 'SENIOR COMPUTER OPERATOR', 'SENIOR CONSTRUCTION ENGINEER', 'SENIOR CONSTRUCTION ESTIMATOR', 'SENIOR CONSTRUCTION INSPECTOR', 'SENIOR CUSTODIAN', 'SENIOR DATA PROCESSING TECHNICIAN', 'SENIOR DETENTION OFFICER', 'SENIOR ELECTRIC SERVICE REPRESENTATIVE', 'SENIOR ELECTRIC TROUBLE DISPATCHER', 'SENIOR ELECTRICAL ENGINEERING DRAFTING TECHNICIAN', 'SENIOR ELECTRICAL INSPECTOR', 'SENIOR ELECTRICAL MECHANIC', 'SENIOR ELECTRICAL MECHANIC SUPERVISOR', 'SENIOR ELECTRICAL REPAIR SUPERVISOR', 'SENIOR ELECTRICAL TEST TECHNICIAN', 'SENIOR ELECTRICIAN', 'SENIOR ENVIRONMENTAL COMPLIANCE INSPECTOR', 'SENIOR ENVIRONMENTAL ENGINEER', 'SENIOR EQUIPMENT MECHANIC', 'WATER UTILITY SUPERINTENDENT', 'SENIOR FIRE PROTECTION ENGINEER', 'SENIOR FORENSIC PRINT SPECIALIST', 'SENIOR GARDENER', 'SENIOR HEATING AND REFRIGERATION INSPECTOR', 'SENIOR HEAVY DUTY EQUIPMENT MECHANIC', 'SENIOR HOUSING INSPECTOR', 'SENIOR HOUSING INVESTIGATOR', 'SENIOR HYDROGRAPHER', 'SENIOR INDUSTRIAL HYGIENIST', 'SENIOR LABOR RELATIONS SPECIALIST', 'SENIOR LEGISLATIVE ASSISTANT', 'SENIOR LIBRARIAN', 'SENIOR LOAD DISPATCHER', 'SENIOR MACHINIST SUPERVISOR', 'SENIOR MANAGEMENT ANALYST', 'SENIOR MECHANICAL ENGINEERING DRAFTING TECHNICIAN', 'SENIOR MECHANICAL REPAIRER', 'SENIOR PAINTER', 'SENIOR PARK MAINTENANCE SUPERVISOR', 'SENIOR PARK RANGER', 'SENIOR PARK SERVICES ATTENDANT', 'SENIOR PARKING ATTENDANT', 'SENIOR PERSONNEL ANALYST', 'SENIOR PHOTOGRAPHER', 'SENIOR PLUMBER', 'SENIOR PLUMBING INSPECTOR', 'SENIOR POLICE SERVICE REPRESENTATIVE', 'SENIOR PORT ELECTRICAL MECHANIC', 'SENIOR PROPERTY OFFICER', 'SENIOR REAL ESTATE OFFICER', 'SENIOR RECREATION DIRECTOR', 'SENIOR ROOFER', 'SENIOR SAFETY ENGINEER ELEVATORS', 'SENIOR SAFETY ENGINEER PRESSURE VESSELS', 'SENIOR SECURITY OFFICER', 'SENIOR STOREKEEPER', 'SENIOR STREET LIGHTING ENGINEER', 'SENIOR STRUCTURAL ENGINEER', 'SENIOR SURVEY SUPERVISOR', 'SENIOR SYSTEMS ANALYST', 'SENIOR TAX AUDITOR', 'SENIOR TITLE EXAMINER', 'SENIOR TRAFFIC SUPERVISOR', 'SENIOR TRANSPORTATION ENGINEER', 'SENIOR TRANSPORTATION INVESTIGATOR', 'SENIOR UNDERGROUND DISTRIBUTION CONSTRUCTION SUPERVISOR', 'SENIOR UTILITY ACCOUNTANT', 'SENIOR UTILITY BUYER', 'SENIOR UTILITY SERVICES SPECIALIST', 'SENIOR UTILITY SERVICES SPECIALIST', 'SENIOR WASTEWATER TREATMENT OPERATOR', 'SENIOR WINDOW CLEANER', "SENIOR WORKERS' COMPENSATION ANALYST", 'SHEET METAL SUPERVISOR', 'SHEET METAL WORKER', 'SHIFT SUPERINTENDENT WASTEWATER TREATMENT', 'SHIP CARPENTER', 'SHOPS SUPERINTENDENT', 'SIGN PAINTER', 'SIGN SHOP SUPERVISOR', 'SIGNAL SYSTEMS ELECTRICIAN', 'SIGNAL SYSTEMS SUPERINTENDENT', 'SIGNAL SYSTEMS SUPERVISOR', 'SOCIAL WORKER', 'SOLID RESOURCES SUPERINTENDENT', 'SOLID WASTE DISPOSAL SUPERINTENDENT', 'SPECIAL INVESTIGATOR', 'SENIOR CRIME AND INTELLIGENCE ANALYST', 'STAFF ASSISTANT TO GENERAL MANAGER WATER AND POWER', 'STEAM PLANT ASSISTANT', 'STEAM PLANT MAINTENANCE MECHANIC', 'STEAM PLANT MAINTENANCE SUPERVISOR', 'STEAM PLANT OPERATOR', 'STOREKEEPER', 'STORES SUPERVISOR', 'STREET LIGHTING CONSTRUCTION AND MAINTENANCE SUPERINTENDENT', 'STREET LIGHTING ELECTRICIAN', 'STREET LIGHTING ELECTRICIAN SUPERVISOR', 'STREET LIGHTING ENGINEER', 'CAMPUS INTERVIEWS ONLY', 'STREET SERVICES GENERAL SUPERINTENDENT', 'STREET SERVICES INVESTIGATOR', 'STREET SERVICES SUPERINTENDENT', 'STREET SERVICES SUPERVISOR', 'STREET SERVICES WORKER', 'STREET TREE SUPERINTENDENT', 'STRUCTURAL ENGINEER', 'STRUCTURAL ENGINEERING ASSOCIATE', 'STRUCTURAL STEEL FABRICATOR', 'STRUCTURAL STEEL FABRICATOR SUPERVISOR', 'SUPERINTENDENT OF RECREATION AND PARKS OPERATIONS', 'SUPERVISING CRIMINALIST', 'SUPERVISING OCCUPATIONAL HEALTH NURSE', 'SUPERVISING TRANSPORTATION PLANNER', 'SUPERVISING WATER SERVICE REPRESENTATIVE', 'SUPPLY SERVICES MANAGER', 'SUPPLY SERVICES PAYMENT CLERK', 'SURVEY PARTY CHIEF', 'SURVEY SUPERVISOR', 'SYSTEMS AIDE', 'SYSTEMS ANALYST', 'SYSTEMS PROGRAMMER', 'TAX AUDITOR', 'TAX COMPLIANCE AIDE', 'TAX COMPLIANCE OFFICER', 'TELECOMMUNICATIONS PLANNING AND UTILIZATION OFFICER', 'Tile Setter', 'TIRE REPAIRER', 'TITLE EXAMINER', 'TRAFFIC MARKING AND SIGN SUPERINTENDENT', 'TRAFFIC OFFICER', 'TRAFFIC PAINTER AND SIGN POSTER', 'TRANSMISSION AND DISTRIBUTION DISTRICT SUPERVISOR', 'TRANSPORTATION ENGINEER', 'TRANSPORTATION ENGINEERING AIDE', 'TRANSPORTATION ENGINEERING ASSOCIATE', 'TRANSPORTATION INVESTIGATOR', 'TRANSPORTATION PLANNING ASSOCIATE', 'TREASURY ACCOUNTANT', 'TREE SURGEON', 'TREE SURGEON ASSISTANT', 'TREE SURGEON SUPERVISOR', 'TRUCK AND EQUIPMENT DISPATCHER', 'TRUCK OPERATOR', 'UNDERGROUND DISTRIBUTION CONSTRUCTION MECHANIC', 'UNDERGROUND DISTRIBUTION CONSTRUCTION SUPERVISOR', 'UPHOLSTERER', 'UTILITIES SERVICE INVESTIGATOR', 'UTILITY ACCOUNTANT', 'UTILITY ADMINISTRATOR', 'UTILITY BUYER', 'UTILITY EXECUTIVE SECRETARY', 'UTILITY SERVICES MANAGER', 'UTILITY SERVICES SPECIALIST', 'VETERINARY TECHNICIAN', 'VIDEO PRODUCTION COORDINATOR', 'VIDEO TECHNICIAN', 'DEPARTMENT OF PUBLIC WORKS', 'VOLUNTEER COORDINATOR', 'WAREHOUSE AND TOOLROOM WORKER', 'WASTEWATER TREATMENT OPERATOR', 'WASTEWATER COLLECTION WORKER', 'WASTEWATER TREATMENT ELECTRICIAN', 'WASTEWATER TREATMENT ELECTRICIAN SUPERVISOR', 'WASTEWATER TREATMENT LABORATORY MANAGER', 'WASTEWATER TREATMENT MECHANIC', 'WASTEWATER TREATMENT MECHANIC SUPERVISOR', 'WASTEWATER TREATMENT OPERATOR', 'WATER BIOLOGIST', 'WATER MICROBIOLOGIST', 'WATER SERVICE REPRESENTATIVE', 'WATER SERVICE SUPERVISOR', 'WATER SERVICE WORKER', 'WATER SERVICES MANAGER', 'WATER TREATMENT OPERATOR', 'WATER TREATMENT SUPERVISOR', 'WATER UTILITY OPERATOR', 'WATER UTILITY OPERATOR SUPERVISOR', 'WATER UTILITY SUPERINTENDENT', 'WATER UTILITY SUPERVISOR', 'WATER UTILITY WORKER', 'WATERSHED RESOURCES SPECIALIST', 'WATERWORKS ENGINEER', 'WATERWORKS MECHANIC SUPERVISOR', 'WELDER', 'WELDER SUPERVISOR', 'WHARFINGER', 'WINDOW CLEANER', "WORKERS' COMPENSATION ANALYST", "WORKERS' COMPENSATION CLAIMS ASSISTANT", 'X-Ray and Laboratory Technician', 'ZOO CURATOR', 'ZOO CURATOR OF EDUCATION', 'ZOO REGISTRAR']



                   

        def jobClassAltFunct(setNo):

            jobClassList=[]

            wordList= setNo.split(" ")

            if "experience as a" in setNo:

                for i in range(0,len(wordList)):

                    if "as" in wordList[i] and "experience" in wordList[i-1]:

                        jobClassList.append(" ".join(wordList[i+2:]))

                        #jobClassList[-1]=jobClassList[-1].split(" or ")

#paid...experience

            elif "paid" in setNo and "experience" in setNo and "experience as" not in setNo:

                for i in range(0,len(wordList)):

                    if "paid" in wordList[i] and "experience" in wordList[i+2]:

                        jobClassList.append(" ".join(wordList[i+1:]))

                    elif "paid" in wordList[i] and "experience" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i+1:]))

                    elif "paid" in wordList[i] and "experience" in wordList[i+3]:

                        jobClassList.append(" ".join(wordList[i+1:]))

                    elif "paid" in wordList[i] and "experience" in wordList[-1]:

                        jobClassList.append(" ".join(wordList[i+1:]))

                    elif len(wordList)>20:    

                        if "paid" in wordList[i] and "experience" in wordList[i+4]:

                            jobClassList.append(" ".join(wordList[i+1:]))

#experience with the City of Los Angeles as a Management Assistant

            elif "experience with the City of Los Angeles as" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and len(wordList)> i+7 and "as" in wordList[i+7]:

                        if len(wordList)>i+8:

                            jobClassList.append(" ".join(wordList[i+9:]))

                        else:

                            jobClassList.append(" ".join(wordList[i+8:]))                   

#experience at the level of Engineering Associate

            elif "experience at the level of" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "of" in wordList[i+4]:

                        jobClassList.append(" ".join(wordList[i+5:]))

                        #jobClassList[-1]=jobClassList[-1].split(" or ")

#experience in a position providing

            elif "experience in a position providing" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "providing" in wordList[i+4]:

                        jobClassList.append(" ".join(wordList[i+5:]))

#full-time paid fire suppression certified experience

            elif "full-time paid" in setNo and "certified experience" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "paid" in wordList[i-4]:

                        jobClassList.append(" ".join(wordList[i-3:]))

#paid experience in the

            elif "paid experience in the" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "the" in wordList[i+2]:

                        jobClassList.append(" ".join(wordList[i+3:]))

#paid experience in

            elif "paid experience in" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "the" not in wordList[i+2]:

                        jobClassList.append(" ".join(wordList[i+2:]))

#paid experience performing 

            elif "paid experience performing" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "performing" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i+2:]))

#paid professional experience in

            elif "paid professional experience in" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "professional" in wordList[i-1]:

                        jobClassList.append(" ".join(wordList[i+2:]))

            elif "office clerical experience" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "office" in wordList[i-2]:

                        jobClassList.append(" ".join(wordList[i-2:]))

#paid experience with the City:

            elif "paid experience with" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "with" in wordList[i+1] and "as" != wordList[i+7]:

                        jobClassList.append(" ".join(wordList[i:]))

#paid experience at

            elif "paid experience at" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "at" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i:]))

#paid experience assisting

            elif "paid experience assisting" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "assisting" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i:]))

#professional experience

            elif "professional experience" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "professional" in wordList[i-1]:

                        jobClassList.append(" ".join(wordList[i-1:]))

#experience working

            elif "experience working" in setNo:

                for i in range(0,len(wordList)):

                    if "experience" in wordList[i] and "working" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i:]))

            elif "hours) as a" in setNo:

                for i in range(0,len(wordList)):

                    if "hours)" in wordList[i] and "as" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i+3:]))

#engineering experience in

            elif "engineering experience in" in setNo:

               for i in range(0,len(wordList)):

                    if "engineering" in wordList[i] and "in" in wordList[i+2]:

                        jobClassList.append(" ".join(wordList[i:]))

            elif "years as a" in setNo:

                for i in range(0,len(wordList)):

                    if "years" in wordList[i] and "as" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i+3:]))

#paid professional engineering

            elif "paid professional" in setNo:

                for i in range(0,len(wordList)):

                    if "paid" in wordList[i] and "professional" in wordList[i+1]:

                        jobClassList.append(" ".join(wordList[i+1:]))

            elif "hours with the City of Los Angeles as" in setNo:

                for i in range(0,len(wordList)):

                    if "hours" in wordList[i] and "Angeles" in wordList[i+6]:

                        jobClassList.append(" ".join(wordList[i+8:]))

            return "/ ".join(jobClassList).replace("####OR####","").replace("####AND####","").strip().strip(",;")

       

        

        def jobClass(setNo):

            jobClasses=[]

            for element in jobTitles:

                if element.lower() in jobClassAltFunct(setNo):

                   jobClasses.append(element)

                elif element.lower().capitalize() in jobClassAltFunct(setNo):

                   jobClasses.append(element)

            if "with the City" in jobClassAltFunct(setNo) and not "at the level of " in jobClassAltFunct(setNo):

                if "experience with the City of Los Angeles as a" in jobClassAltFunct(setNo):

                   jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("with the City")+34:])

                elif " with the City" in jobClassAltFunct(setNo) and not "xperience with the City" in jobClassAltFunct(setNo):

                   jobClasses.append(jobClassAltFunct(setNo)[:jobClassAltFunct(setNo).index(" with the City")])

            if "at the level of " in jobClassAltFunct(setNo):

                if " in the area" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(" in the area")])

                elif ", in building" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(", in building")])                

                elif " in the administration" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(" in the administration ")])

                elif " checking" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(" checking")])

                elif ", two years of" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(", two years of")])

                elif " which provides" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(" which provides")])

                elif " performing" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(" performing")])

                elif " in institutional" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(" in institutional")])

                elif " developing" in jobClassAltFunct(setNo):

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:jobClassAltFunct(setNo).index(" developing")])



                else:

                    jobClasses.append(jobClassAltFunct(setNo)[jobClassAltFunct(setNo).index("at the level of ")+16:])

            elif "with the Los Angeles" in jobClassAltFunct(setNo):

                jobClasses.append(jobClassAltFunct(setNo)[:jobClassAltFunct(setNo).index(" with the Los")])       

            elif " or in a class" in jobClassAltFunct(setNo):

                if len(jobClasses)>0 and " or in a class" in jobClasses[-1]:

                    jobClasses[-1]=jobClasses[-1][:jobClasses[-1].index(" or in a class")].strip(",")

                elif " or in a class" not in " ".join(jobClasses):

                    jobClasses.append(jobClassAltFunct(setNo)[:jobClassAltFunct(setNo).index(" or in a class")].strip(",")) 

            elif " or in a position" in jobClassAltFunct(setNo):

                if len(jobClasses)>0 and " or in a position" in jobClasses[-1]:

                    jobClasses[-1]=jobClasses[-1][:jobClasses[-1].index(" or in a position")].strip(",")

                elif " or in a position" not in " ".join(jobClasses):

                    jobClasses.append(jobClassAltFunct(setNo)[:jobClassAltFunct(setNo).index(" or in a position")].strip(","))

            for j in range(0,len(jobClasses)):

                if " with" in jobClasses[j]:

                    jobClasses[j]=jobClasses[j][:jobClasses[j].index(" with")].strip().strip(",;")

                if " as a" in jobClasses[j]:

                       jobClasses[j]=jobClasses[j][jobClasses[j].index(" as a")+6:].strip().strip(",;")

                if " interpreting" in jobClasses[j]:

                    jobClasses[j]=jobClasses[j][:jobClasses[j].index(" interpreting")].strip().strip(",;")

                if " coordinating" in jobClasses[j]:

                    jobClasses[j]=jobClasses[j][:jobClasses[j].index(" coordinating")].strip().strip(",;")

                if "experience" == jobClasses[j]:

                   jobClasses[j] = ""

                for k in range(0,len(jobClasses)):

                   if jobClasses[k].upper() == jobClasses[j].upper() and j!=k:

                       jobClasses[k]=""

                   #if jobClasses[k].upper() in jobClasses[j].upper() and jobClasses[j]!=jobClasses[k]:

                       #jobClasses[k]=""

                       

            return "/ ".join(jobClasses).upper().strip().strip("/")

                   

        self.jobClass1=jobClass(self.requirementString1)           

        self.jobClass2=jobClass(self.requirementString2)           

        self.jobClass3=jobClass(self.requirementString3)

        self.jobClass4=jobClass(self.requirementString4)           

        self.jobClass5=jobClass(self.requirementString5)           

        self.jobClass6=jobClass(self.requirementString6)           

        self.jobClass7=jobClass(self.requirementString7)           

        self.jobClass8=jobClass(self.requirementString8)           

        



        self.jobClassList1=jobClassAltFunct(self.requirementString1)

        self.jobClassList2=jobClassAltFunct(self.requirementString2)

        self.jobClassList3=jobClassAltFunct(self.requirementString3)

        self.jobClassList4=jobClassAltFunct(self.requirementString4)

        self.jobClassList5=jobClassAltFunct(self.requirementString5)

        self.jobClassList6=jobClassAltFunct(self.requirementString6)

        self.jobClassList7=jobClassAltFunct(self.requirementString7)

        self.jobClassList8=jobClassAltFunct(self.requirementString8)





#--------------------------------------------SCHOOL_TYPE----------------------------------------------------

        def schoolType(setNo):

            schoolTypeList=["","","","","","",""]

            if "college" in setNo or "College" in setNo:

                schoolTypeList[0]="COLLEGE"

            if "university" in setNo or "University" in setNo:

                schoolTypeList[1]="UNIVERSITY"

            if "trade school" in setNo or "Trade school" in setNo:

                schoolTypeList[2]="TRADE SCHOOL"

            if "technical school" in setNo or "Technical school" in setNo:

                schoolTypeList[3]="TECHNICAL SCHOOL"

            if "trade or technical school" in setNo:

                schoolTypeList[2]="TRADE SCHOOL"

                schoolTypeList[3]="TECHNICAL SCHOOL"

            if "high school" in setNo or "High School" in setNo:

                schoolTypeList[4]="HIGH SCHOOL"

            if "G.E.D." in setNo:

                schoolTypeList[5]="G.E.D."

            if "apprenticeship" in setNo or "Apprenticeship" in setNo:

                schoolTypeList[6]="APPRENTICESHIP"

            schoolTypeList="/".join(schoolTypeList)

            schoolTypeList=schoolTypeList.replace("//////"," ").replace("/////","/").replace("////","/").replace("///","/").replace("//","/")

            if schoolTypeList[-1]=="/" or schoolTypeList[-1]==" ":

                schoolTypeList=schoolTypeList[:-1]

            if schoolTypeList!="" and schoolTypeList[0]=="/":

                schoolTypeList=schoolTypeList[1:]

            return schoolTypeList.strip()

                

        self.schoolType1=schoolType(self.requirementString1)

        self.schoolType2=schoolType(self.requirementString2)

        self.schoolType3=schoolType(self.requirementString3)

        self.schoolType4=schoolType(self.requirementString4)

        self.schoolType5=schoolType(self.requirementString5)

        self.schoolType6=schoolType(self.requirementString6)

        self.schoolType7=schoolType(self.requirementString7)

        self.schoolType8=schoolType(self.requirementString8)               

            

#-------------------------------COURSE_LENGTH---------------------------------------------------

        numberDict={"one":"1", "two":"2","three":"3", "four":"4","five":"5","six":"6","seven":"7","eight":"8",

                    "nine":"9","ten":"10","eleven":"11","twelve":"12","One":"1", "Two":"2","Three":"3", "Four":"4",

                    "Five":"5","Six":"6","Seven":"7","Eight":"8",

                    "Nine":"9","Ten":"10","Eleven":"11","Twelve":"12"}

        def courseLen(setNo):

            courseLenList=[]

            wordList=setNo.split(" ")

            for i in range(0,len(wordList)):

                if "semester" in wordList[i] or "Semester" in wordList[i]:

                    if wordList[i-1] in numberDict:

                        courseLenList.append(numberDict[wordList[i-1]]+"S")

                    else:

                        courseLenList.append(wordList[i-1]+"S")

                if "quarter" in wordList[i] and "unit" in wordList[i+1]:

                    if wordList[i-1] in numberDict:

                        courseLenList.append(numberDict[wordList[i-1]]+"Q")

                    else:

                        courseLenList.append(wordList[i-1]+"Q")

                if "quarters" in wordList[i] and courseLenList!=[]:

                    if wordList[i-1] in numberDict:

                        courseLenList.append(numberDict[wordList[i-1]]+"Q")

                    else:

                        courseLenList.append(wordList[i-1]+"Q")

            for j in range(0,len(courseLenList)):

                courseLenList[j]=courseLenList[j].replace("(","")

            return "|".join(courseLenList)

        

        self.courseLen1=courseLen(self.requirementString1)

        self.courseLen2=courseLen(self.requirementString2)

        self.courseLen3=courseLen(self.requirementString3)

        self.courseLen4=courseLen(self.requirementString4)

        self.courseLen5=courseLen(self.requirementString5)

        self.courseLen6=courseLen(self.requirementString6)

        self.courseLen7=courseLen(self.requirementString7)

        self.courseLen8=courseLen(self.requirementString8)

    

#----------------------------COURSE_SUBJECT----------------------------------------------



        courseAllSubList=['ecology', 'plane surveying', 'mass communication', 'Finance/Accounting', 'communications'

               , 'electronics', 'Electrical Engineering', 'Computer Aided Drafting  and  Design (CADD)'

               , 'environmental impact analysis', 'algebra', 'information systems', 'trigonometry', 'accounting'

               , 'marine biology', 'environmental law', 'laboratory module', 'environmental planning'

               , 'water quality', 'mechanical engineering', 'general chemistry', 'environmental engineering'

               , 'upper-division mathematics', 'control systems', 'mathematics', 'environmental health'

               , 'Geometry', 'geographic information systems', 'biology', 'structural engineering', 'microbiology'

               , 'oceanography', 'real estate', 'Trigonometry', 'botany', 'Public Administration'

               , 'civil engineering', 'air quality', 'geography', 'industrial hygiene', 'geometry', 'zoology'

               , 'electrical or civil engineering', 'toxicology', 'public relations', 'geology', 'field biology'

               , 'computer science', 'environmental auditing', 'computer aided drafting','communications engineering'

               , 'Computer Aided Drafting Design (CADD)', 'regulatory oversight', 'statistics'

               , 'transportation engineering', 'electrical engineering', 'environmental policy'

               , 'journalism', 'biochemistry', 'Chemical or Environmental Engineering', 'social sciences'

               , 'groundwater and surface water systems', 'drafting', 'chemistry', 'Supply Chain Management'

               , 'limnology', 'computer engineering', 'Business Administration',"architectural drafting"

               ,"design utilizing CADD systems", "electronics field of concentration"

               ,"military electronics technician course", "business administration", "electrical or electronics"

               ,"finance", "industrial electronics", "industrial electricity","environmental science"

               ,"solid waste management technology","water supply technology"

               ,"stormwater or wastewater treatment technology","engineering", "horticulture", "writing"

               ,"English", "Real Estate Principles", "Real Estate Finance", "Real Estate Appraisal"

               ,"Real Estate Law", "real estate", "finance", "business","physics","electricity","Microbiology"

               ,"Bacteriology", "Parasitology", "Virology", "Microbial Ecology", "Microbial Physiology"

               ,"Molecular Biology", "Mycology", "Biochemistry", "Public Health", "Statistics","telecommunications"

               ,"advanced statistics", "research design", "psychological measurement", "construction"

               ,"design","building inspection technology","construction inspection","architectural drafting"

               ,"civil, mechanical, or electrical engineering technology","public works construction"

               ,"civil, mechanical, electrical, or fire protection engineering technology","math", ]

        def courseSub(setNo):

            courseSubList=[]

            if "emester" in setNo:

                courseSnip=setNo[setNo.index("emester"):]

            if "courses:" in setNo:

                courseSnip=setNo[setNo.index("courses:"):]

            for element in courseAllSubList:

                if "emester" in setNo:

                   if courseLen(setNo)!=[] and element in setNo[setNo.index("emester"):]:

                       courseSubList.append(element)

                if courseLen(setNo)!=[] and "courses:" in setNo:

                    if element in setNo[setNo.index("courses:"):]:

                        courseSubList.append(element)

            removeList=[]       

            for i in range(0,len(courseSubList)):

                for j in range(0,len(courseSubList)):

                    if courseSubList[j] in courseSubList[i] and courseSubList[j]!=courseSubList[i]:

                        if courseSubList[j] not in courseSnip[courseSnip.index(courseSubList[i])+len(courseSubList[i]):] and courseSubList[j] not in courseSnip[:courseSnip.index(courseSubList[i])]:

                            removeList.append(courseSubList[j])

                    if courseSubList[j] == courseSubList[i] and i!=j:

                        courseSubList[j]=" "

                        removeList.append(" ")

                            

            for element in removeList:

                if element in courseSubList:

                    courseSubList.remove(element)

        

            for k in range(0,len(courseSubList)):

                courseSubList[k].upper()

              

            return "/ ".join(courseSubList).upper()

        

        self.courseSubjects1=courseSub(self.requirementString1)

        self.courseSubjects2=courseSub(self.requirementString2)

        self.courseSubjects3=courseSub(self.requirementString3)

        self.courseSubjects4=courseSub(self.requirementString4)

        self.courseSubjects5=courseSub(self.requirementString5)

        self.courseSubjects6=courseSub(self.requirementString6)

        self.courseSubjects7=courseSub(self.requirementString7)

        self.courseSubjects8=courseSub(self.requirementString8)

            

#--------------------------------------------------EDUCATION_MAJOR------------------------------------------



        def eduMajor(setNo):

            eduMajorString=""

            if "major in " in setNo:

                eduMajorString=setNo[setNo.index("major in")+9:].strip()

            if "majoring in " in setNo:

                eduMajorString=setNo[setNo.index("majoring in")+12:].strip()

            if "degree in " in setNo:

                eduMajorString=setNo[setNo.index("degree in")+10:].strip()

            

            eduMajorString=eduMajorString.replace("; ####AND####","").replace("; ####OR####", "")

            eduMajorWords=eduMajorString.split(" ")

            for i in range(0,len(eduMajorWords)):

                if len(eduMajorWords)>= i+2:

                    if "and" == eduMajorWords[i-2] and "year" in eduMajorWords[i]:

                        eduMajorString=" ".join(eduMajorWords[:i-2])

                        break

            if ", which" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index(", which")]

            if ", including" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index(", including")]

            if ", or upon" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index(", or upon")]

            if "from an accredited" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index("from an accredited")]

            if "and successful" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index("and successful")]

            if "with at least" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index("with at least")]

            if "; and" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index("; and")]

            eduMajorString=eduMajorString.strip(",").strip(";")

            if " b. " in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index(" b. ")]

            if "from a school" in eduMajorString:

                eduMajorString=eduMajorString[:eduMajorString.index("from a school")]

            eduMajorString=eduMajorString.strip(",").strip(";")

             

            return eduMajorString.upper()

        

        self.eduMajor1=eduMajor(self.requirementString1)

        self.eduMajor2=eduMajor(self.requirementString2)

        self.eduMajor3=eduMajor(self.requirementString3)

        self.eduMajor4=eduMajor(self.requirementString4)

        self.eduMajor5=eduMajor(self.requirementString5)

        self.eduMajor6=eduMajor(self.requirementString6)

        self.eduMajor7=eduMajor(self.requirementString7)

        self.eduMajor8=eduMajor(self.requirementString8)





#--------------------------------------------EDUCATION_YEARS---------------------------------------

        numberDict={"one":"1", "two":"2","three":"3", "four":"4","five":"5","six":"6","seven":"7","eight":"8",

                    "nine":"9","ten":"10","eleven":"11","twelve":"12","eighteen":"18","One":"1", "Two":"2","Three":"3", "Four":"4",

                    "Five":"5","Six":"6","Seven":"7","Eight":"8",

                    "Nine":"9","Ten":"10","Eleven":"11","Twelve":"12", "Eighteen":"18"}

        def eduYear(setNo):

            eduYearStr=""

            eduMonthStr=""

            eduYearInt=-99

            eduMonthInt=-99

            if schoolType(setNo)!="":

                if "year college" in setNo:

                    eduYearStr=setNo[setNo.index("year college")-10:setNo.index("year college")]

                if "year apprenticeship" in setNo:

                    eduYearStr=setNo[setNo.index("year apprenticeship")-10:setNo.index("year apprenticeship")]

                if schoolType(setNo)=="APPRENTICESHIP":

                   if "year" in setNo:

                       eduYearStr=setNo[setNo.index("year")-10:setNo.index("year")]

                   if "month" in setNo:

                       eduMonthStr=setNo[setNo.index("month")-10:setNo.index("month")]

            for element in numberDict:

                if element in eduMonthStr:

                    eduMonthInt=int(numberDict[element])

                   

            for element in numberDict:

                if element in eduYearStr:

                    eduYearInt=int(numberDict[element])

            if eduMonthInt!=-99:

                   eduYearInt+= eduMonthInt/12

            return eduYearInt

                

        self.eduYear1=eduYear(self.requirementString1)

        self.eduYear2=eduYear(self.requirementString2)

        self.eduYear3=eduYear(self.requirementString3)

        self.eduYear4=eduYear(self.requirementString4)

        self.eduYear5=eduYear(self.requirementString5)

        self.eduYear6=eduYear(self.requirementString6)

        self.eduYear7=eduYear(self.requirementString7)

        self.eduYear8=eduYear(self.requirementString8)

        

import csv



with open ("job_bulletins.csv", "w") as csv_file:

    fieldnames = ["FILE_NAME", "JOB_CLASS_TITLE", "JOB_CLASS_NO", "REQUIREMENT_SET",

                  "REQUIREMENT_SET_ID1","REQUIREMENT_SUBSET_ID1","EDUCATION_YEARS1","SCHOOL_TYPE1","EDUCATION_MAJOR1","EXPERIENCE_LENGTH1","FULL_TIME_PART_TIME1","EXP_JOB_CLASS_TITLE1","EXP_JOB_ADD_INFO1","COURSE_LENGTH1","COURSE_SUBJECT1",

                  "REQUIREMENT_SET_ID2","REQUIREMENT_SUBSET_ID2","EDUCATION_YEARS2","SCHOOL_TYPE2","EDUCATION_MAJOR2","EXPERIENCE_LENGTH2","FULL_TIME_PART_TIME2","EXP_JOB_CLASS_TITLE2","EXP_JOB_ADD_INFO2","COURSE_LENGTH2","COURSE_SUBJECT2",

                  "REQUIREMENT_SET_ID3","REQUIREMENT_SUBSET_ID3","EDUCATION_YEARS3","SCHOOL_TYPE3","EDUCATION_MAJOR3","EXPERIENCE_LENGTH3","FULL_TIME_PART_TIME3","EXP_JOB_CLASS_TITLE3","EXP_JOB_ADD_INFO3","COURSE_LENGTH3","COURSE_SUBJECT3",

                  "REQUIREMENT_SET_ID4","REQUIREMENT_SUBSET_ID4","EDUCATION_YEARS4","SCHOOL_TYPE4","EDUCATION_MAJOR4","EXPERIENCE_LENGTH4","FULL_TIME_PART_TIME4","EXP_JOB_CLASS_TITLE4","EXP_JOB_ADD_INFO4","COURSE_LENGTH4","COURSE_SUBJECT4",

                  "REQUIREMENT_SET_ID5","REQUIREMENT_SUBSET_ID5","EDUCATION_YEARS5","SCHOOL_TYPE5","EDUCATION_MAJOR5","EXPERIENCE_LENGTH5","FULL_TIME_PART_TIME5","EXP_JOB_CLASS_TITLE5","EXP_JOB_ADD_INFO5","COURSE_LENGTH5","COURSE_SUBJECT5",

                  "REQUIREMENT_SET_ID6","REQUIREMENT_SUBSET_ID6","EDUCATION_YEARS6","SCHOOL_TYPE6","EDUCATION_MAJOR6","EXPERIENCE_LENGTH6","FULL_TIME_PART_TIME6","EXP_JOB_CLASS_TITLE6","EXP_JOB_ADD_INFO6","COURSE_LENGTH6","COURSE_SUBJECT6",

                  "REQUIREMENT_SET_ID7","REQUIREMENT_SUBSET_ID7","EDUCATION_YEARS7","SCHOOL_TYPE7","EDUCATION_MAJOR7","EXPERIENCE_LENGTH7","FULL_TIME_PART_TIME7","EXP_JOB_CLASS_TITLE7","EXP_JOB_ADD_INFO7","COURSE_LENGTH7","COURSE_SUBJECT7",

                  "REQUIREMENT_SET_ID8","REQUIREMENT_SUBSET_ID8","EDUCATION_YEARS8","SCHOOL_TYPE8","EDUCATION_MAJOR8","EXPERIENCE_LENGTH8","FULL_TIME_PART_TIME8","EXP_JOB_CLASS_TITLE8","EXP_JOB_ADD_INFO8","COURSE_LENGTH8","COURSE_SUBJECT8",

                  "JOB_DUTIES",            

                  "DRIVERS_LICENSE_REQ","DRIV_LIC_TYPE",

                  "ADDTL_LIC",

                  "EXAM_TYPE","OPEN","INT_DEPT_PROM","DEPT_PROM","OPEN_INT_PROM",

                  "ENTRY_SALARY_GEN","ENTRY_SALARY_DWP", "ENTRY_SALARY_AIRPRT","ENTRY_SALARY_HRBR",

                  "OPEN_DATE"]

    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

    

    writer.writeheader()

    for file in fileList:

        r = Reader("../input/cityofla/CityofLA/Job Bulletins", file)

        r.read()

        writer.writerow({"FILE_NAME": file,

                         "JOB_CLASS_TITLE": r.jobTitle, "JOB_CLASS_NO": r.jobNo

                        ,"REQUIREMENT_SET": r.requirementString

                        ,"REQUIREMENT_SET_ID1": r.requirementString1 

                        ,"REQUIREMENT_SET_ID2": r.requirementString2,"REQUIREMENT_SET_ID3": r.requirementString3 

                        ,"REQUIREMENT_SET_ID4": r.requirementString4,"REQUIREMENT_SET_ID5": r.requirementString5 

                        ,"REQUIREMENT_SET_ID6": r.requirementString6,"REQUIREMENT_SET_ID7": r.requirementString7

                        ,"REQUIREMENT_SET_ID8": r.requirementString8,"REQUIREMENT_SUBSET_ID1": r.requirementSub1

                        ,"REQUIREMENT_SUBSET_ID1": r.requirementSub1,"REQUIREMENT_SUBSET_ID2": r.requirementSub2

                        ,"REQUIREMENT_SUBSET_ID3": r.requirementSub3,"REQUIREMENT_SUBSET_ID4": r.requirementSub4

                        ,"REQUIREMENT_SUBSET_ID5": r.requirementSub5,"REQUIREMENT_SUBSET_ID6": r.requirementSub6

                        ,"REQUIREMENT_SUBSET_ID7": r.requirementSub7,"REQUIREMENT_SUBSET_ID8": r.requirementSub8

                        ,"EDUCATION_YEARS1":r.eduYear1,"EDUCATION_YEARS2":r.eduYear2,"EDUCATION_YEARS3":r.eduYear3

                        ,"EDUCATION_YEARS4":r.eduYear4,"EDUCATION_YEARS5":r.eduYear5,"EDUCATION_YEARS6":r.eduYear6

                        ,"EDUCATION_YEARS7":r.eduYear7,"EDUCATION_YEARS8":r.eduYear8

                        ,"SCHOOL_TYPE1":r.schoolType1,"SCHOOL_TYPE2":r.schoolType2,"SCHOOL_TYPE3":r.schoolType3

                        ,"SCHOOL_TYPE4":r.schoolType4,"SCHOOL_TYPE5":r.schoolType5,"SCHOOL_TYPE6":r.schoolType6

                        ,"SCHOOL_TYPE7":r.schoolType7,"SCHOOL_TYPE8":r.schoolType8

                        ,"EDUCATION_MAJOR1":r.eduMajor1,"EDUCATION_MAJOR2":r.eduMajor2,"EDUCATION_MAJOR3":r.eduMajor3

                        ,"EDUCATION_MAJOR4":r.eduMajor4,"EDUCATION_MAJOR5":r.eduMajor5,"EDUCATION_MAJOR6":r.eduMajor6

                        ,"EDUCATION_MAJOR7":r.eduMajor7,"EDUCATION_MAJOR8":r.eduMajor8

                        ,"EXPERIENCE_LENGTH1":r.expYearList1,"EXPERIENCE_LENGTH2":r.expYearList2

                        ,"EXPERIENCE_LENGTH3":r.expYearList3,"EXPERIENCE_LENGTH4":r.expYearList4

                        ,"EXPERIENCE_LENGTH5":r.expYearList5,"EXPERIENCE_LENGTH6":r.expYearList6

                        ,"EXPERIENCE_LENGTH7":r.expYearList7,"EXPERIENCE_LENGTH8":r.expYearList8

                        ,"EXP_JOB_CLASS_TITLE1":r.jobClass1,"EXP_JOB_CLASS_TITLE2":r.jobClass2

                        ,"EXP_JOB_CLASS_TITLE3":r.jobClass3,"EXP_JOB_CLASS_TITLE4":r.jobClass4

                        ,"EXP_JOB_CLASS_TITLE5":r.jobClass5,"EXP_JOB_CLASS_TITLE6":r.jobClass6

                        ,"EXP_JOB_CLASS_TITLE7":r.jobClass7,"EXP_JOB_CLASS_TITLE8":r.jobClass8

                        ,"EXP_JOB_ADD_INFO1":r.jobClassList1,"EXP_JOB_ADD_INFO2":r.jobClassList2

                        ,"EXP_JOB_ADD_INFO3":r.jobClassList3,"EXP_JOB_ADD_INFO4":r.jobClassList4

                        ,"EXP_JOB_ADD_INFO5":r.jobClassList5,"EXP_JOB_ADD_INFO6":r.jobClassList6

                        ,"EXP_JOB_ADD_INFO7":r.jobClassList7,"EXP_JOB_ADD_INFO8":r.jobClassList8

                        ,"COURSE_LENGTH1":r.courseLen1,"COURSE_LENGTH2":r.courseLen2,"COURSE_LENGTH3":r.courseLen3

                        ,"COURSE_LENGTH4":r.courseLen4,"COURSE_LENGTH5":r.courseLen5,"COURSE_LENGTH6":r.courseLen6

                        ,"COURSE_LENGTH7":r.courseLen7,"COURSE_LENGTH8":r.courseLen8

                        ,"COURSE_SUBJECT1":r.courseSubjects1,"COURSE_SUBJECT2":r.courseSubjects2

                        ,"COURSE_SUBJECT3":r.courseSubjects3,"COURSE_SUBJECT4":r.courseSubjects4 

                        ,"COURSE_SUBJECT5":r.courseSubjects5,"COURSE_SUBJECT6":r.courseSubjects6

                        ,"COURSE_SUBJECT7":r.courseSubjects7,"COURSE_SUBJECT8":r.courseSubjects8

                        ,"JOB_DUTIES": r.dutiesStr, "FULL_TIME_PART_TIME1":r.fullpart1, "FULL_TIME_PART_TIME2":r.fullpart2

                        ,"FULL_TIME_PART_TIME3":r.fullpart3,"FULL_TIME_PART_TIME4":r.fullpart4,"FULL_TIME_PART_TIME5":r.fullpart5

                        ,"FULL_TIME_PART_TIME6":r.fullpart6,"FULL_TIME_PART_TIME7":r.fullpart7,"FULL_TIME_PART_TIME8":r.fullpart8

                        ,"DRIVERS_LICENSE_REQ": r.driveLic ,"DRIV_LIC_TYPE": r.licType

                        ,"ADDTL_LIC": " | ".join(r.otherLic),"EXAM_TYPE": r.examType

                        ,"OPEN_DATE": r.openDate, "OPEN": r.examOpenTo, "INT_DEPT_PROM": r.intDeptProm

                        ,"OPEN_INT_PROM": r.openIntProm,"DEPT_PROM": r.deptProm,"ENTRY_SALARY_GEN": r.salary_gen

                        ,"ENTRY_SALARY_DWP": r.salary_dwp ,"ENTRY_SALARY_AIRPRT": r.salary_airport

                        ,"ENTRY_SALARY_HRBR": r.salary_harbor})