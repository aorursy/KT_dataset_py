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
!pip install spacy==2.0.11
!pip install "msgpack-numpy<0.4.4.0"
!pip install geograpy3
from __future__ import unicode_literals, print_function
import plac
import random
from pathlib import Path
import ast
from spacy.util import minibatch, compounding
import spacy
import numpy
import geograpy
from nltk.corpus import stopwords
from numpy.testing import assert_almost_equal
from scipy.spatial import distance
import pandas as pd
import ast
import re
from nltk.tokenize import word_tokenize
temp = pd.read_csv("/kaggle/input/naukri-cleaned/naukri_cleaned.csv")
temp = temp.iloc[:, 1:]
temp.head(3)
temp1 = pd.read_csv("/kaggle/input/timesjobpostingdata/times_job_postings_data.csv")
#temp1 = temp1.iloc[:, 1:]
temp1.head(3)
for i in range(len(temp1)):
    temp1['job description'] = temp1['job description'][i].replace('Job Description        ', '')
temp1['job description'][:2]
len(temp1)
#temp1.info()
temp1.dropna(inplace=True)
len(temp1)
temp1.drop_duplicates(keep='first',inplace=True)
print(len(temp1))
temp1.head(3)
list_of_jd = []
for i in temp.iterrows():
    row_text = list(i)
    s = ""
    for j, v in enumerate(row_text[1]):
        if j == 6:
            skills = ", ".join(ast.literal_eval(v))
            v = skills
        s += v + " "
    list_of_jd.append(s)
list_of_jd[:3]
list_of_jd2 = []
for i in temp1.iterrows():
    row_text = list(i)
    s = ""
    for j, v in enumerate(row_text[1]):
        if j == 6:
            skills = ", ".join(ast.literal_eval(v))
            v = skills
        s += str(v) + " "
    list_of_jd2.append(s)
list_of_jd2[1]
def get_spacy_train_format(csv):
    list_of_jd = []
    for i in csv.iterrows():
        row_text = list(i)
        s = ""
        for j, v in enumerate(row_text[1]):
            if j == 6:
                skills = ", ".join(ast.literal_eval(v))
                v = skills
            s += v + " "
        list_of_jd.append(s)
    list_jt = []
    list_cmpn = []
    list_exp = []
    list_salary = []
    list_location = []
    list_education = []
    list_skill = []
    list_i_skill = []

    def remove_nested_parens(input_str):
        """Returns a copy of 'input_str' with any parenthesized text removed. Nested parentheses are handled."""
        result = ''
        paren_level = 0
        for ch in input_str:
            if ch == '(':
                paren_level += 1
            elif (ch == ')') and paren_level:
                paren_level -= 1
            elif not paren_level:
                result += ch
        return result

    c = 0
    for i in temp.iterrows():
        row_text = list(i)
        s = " "
        c += 1
        for j, v in enumerate(row_text[1]):
            if j == 0:
                list_jt.append(v)
            if j == 1:
                list_cmpn.append(v)
            if j == 2:
                list_exp.append(v)
            if j == 3:
                list_salary.append(v)
            if j == 4:
                list_location.append(v)
            if j == 5:
                list_education.append(v)
            if j == 6:
                skills = ", ".join(ast.literal_eval(v))
                list_skill.append(skills)
    for i in range(len(list_education)):
        list_education[i] = list_education[i].replace("Education", "").replace("UG", '')
    list_cmp_without_review = []
    for i in list_cmpn:
        list_cmp_without_review.append(remove_nested_parens(i))
    c = 0
    data = []
    for i, j, k, l, m, n, o in zip(list_of_jd, list_exp, list_jt, list_cmp_without_review, list_location, list_skill,
                                   list_education):
        try:
            match = re.search(j, i)
            s = match.start()
            e = match.end()
            entity_dict = {}
            list_of_index = []
            list_of_index.append(tuple([s, e, "experience"]))
            match = re.search(k, i)
            s = match.start()
            e = match.end()
            list_of_index.append(tuple([s, e, "job titles"]))
            match = re.search(l, i)
            s = match.start()
            e = match.end()
            list_of_index.append(tuple([s, e, "company"]))
            match = re.search(m, i)
            s = match.start()
            e = match.end()
            list_of_index.append(tuple([s, e, "location"]))
            match = re.search(n, i)
            s = match.start()
            e = match.end()
            list_of_index.append(tuple([s, e, "skill"]))
            match = re.search(o, i)
            s = match.start()
            e = match.end()
            list_of_index.append(tuple([s, e, "education"]))
            entity_dict["entities"] = list_of_index
            data.append(tuple([i, entity_dict]))
        except:
            c += 1
    return data
data = get_spacy_train_format(temp)
data[:3]
list_of_jd[:3]
model_path = '/kaggle/input/jobdescnerspacy/ner-model'
nlp = spacy.load(model_path)
if nlp:
    print("Model loaded")
else:
    print("Train again")
!python -m spacy validate
class Entities:
    def __init__(self,text,nlp):
        self.text = text
        self.nlp = nlp
        
    def extract_title(self):
        doc = self.nlp(self.text)
        jt = []
        for ent in doc.ents:
            if ent.label_ == 'job titles':
                if len(ent.text) < 50:
                    for i in ent.text.split('\n'):
                        jt.append(i)
                else:
                    jt.append(ent.text[:27])
        print("Job titles :")
        if len(jt) == 0:
            return None
        else : return jt[0]
    
    def extract_cmpny(self):
        doc = self.nlp(self.text)
        cmp = []
        for ent in doc.ents:
            if ent.label_ == 'company':
                    if len(ent.text) < 50:
                        for i in ent.text.split('\n'):
                            cmp.append(i)
                    else:
                        s = ent.text[:25]
                        for i in ent.text.split('\n'):
                            cmp.append(s)
        print("companies")
        if len(cmp) == 0:
            return None
        else : return cmp[0]
 
    def extract_exp(self):
        doc = self.nlp(self.text)
        exp = []
        for ent in doc.ents:
            if ent.label_ == 'experience':
                    # exp.append(ent)
                    if bool(re.search(r'\d', ent.text)) or bool(re.search('experience', ent.text)):
                        exp.append(ent.text)    
        result = re.findall(r"([\d+-]+)\s+(years?)", self.text, re.IGNORECASE)
        print("experience")
#         for i in result.split('\n')
        print(result)
        return exp
    
    def extract_skill(self):
        doc = self.nlp(self.text)
        skill = []
        print("skills :")
        for ent in doc.ents:
            if ent.label_ == 'skill':
                    skill.append(ent.text)
        return skill
        
    def extract_location(self):
        #nlp = spacy.load('en_core_web_sm')
        # Grad all general stop words
        STOPWORDS = set(stopwords.words('english'))
        # Locations
        cities = ['bangalore','bengaluru', 'mumbai', 'chennai', 'delhi', 'pune', 'hyderabad']
        #doc = nlp(txt)
        #words = [token.text for token in doc if token.is_stop != True and token.is_punct != True]
        words = word_tokenize(self.text)
        loc = set()
        places = geograpy.get_place_context(text=self.text)
        print("location :")
        for i in words:
            if i.lower() in cities:
                loc.add(i)
        for i in places.countries:
            if i.lower() in cities:
                print(i)
        # print(words)
        return loc

    def extract_email(self):
        email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", self.text)
        print("email :")
        if email:
            try:
                return email[0].split()[0].strip(';')
            except IndexError:
                return None
    
    def extract_education(self):
        #nlp = spacy.load('en_core_web_sm')
        # Grad all general stop words
        STOPWORDS = set(stopwords.words('english'))
        # Education Degrees
        EDUCATION = [
        'BE', 'B.E.', 'B.E', 'BS', 'B.S',
        'ME', 'M.E', 'M.E.', 'MS', 'M.S',
        'BTECH', 'B.TECH', 'M.TECH', 'MTECH', "M.B.A", "MBA",
        'SSC', 'HSC', 'CBSE', 'ICSE', 'X', 'XII', 'GRADUATE', "BACHELOR", "BACHELOR'S", "BACHELORS", "MASTER'S", "MASTERS"]
        words = word_tokenize(self.text)
        edu = set()
        print("education :")
        for i in words:
            if i.upper() in EDUCATION:
                edu.add(i)
         # print(words)
        return edu

obj = Entities(text,nlp)
print(obj.extract_title())
print(obj.extract_cmpny())
print(obj.extract_exp())
print(obj.extract_location())
print(obj.extract_education())
print(obj.extract_email())
print(obj.extract_skill())
obj = Entities(text1,nlp)
print(obj.extract_title())
print(obj.extract_cmpny())
print(obj.extract_exp())
print(obj.extract_location())
print(obj.extract_education())
print(obj.extract_email())
print(obj.extract_skill())
obj = Entities(text2,nlp)
print(obj.extract_title())
print(obj.extract_cmpny())
print(obj.extract_exp())
print(obj.extract_location())
print(obj.extract_education())
print(obj.extract_email())
print(obj.extract_skill())
obj = Entities(text3,nlp)
print(obj.extract_title())
print(obj.extract_cmpny())
print(obj.extract_exp())
print(obj.extract_location())
print(obj.extract_education())
print(obj.extract_email())
print(obj.extract_skill())
obj = Entities(text4,nlp)
print(obj.extract_title())
print(obj.extract_cmpny())
print(obj.extract_exp())
print(obj.extract_location())
print(obj.extract_education())
print(obj.extract_email())
print(obj.extract_skill())
obj = Entities(text5,nlp)
print(obj.extract_title())
print(obj.extract_cmpny())
print(obj.extract_exp())
print(obj.extract_location())
print(obj.extract_education())
print(obj.extract_email())
print(obj.extract_skill())
text = """Management Trainee - Publishing
Apr 16, 2020
Genpact

    2 to 3 Yrs
    Mumbai City

    Skills:
    accounts,  finance,  accountingView all

Apply on website
Send me similar jobs
Share
Save
Job Description

With a startup spirit and 90,000+ curious and courageous minds, we have the expertise to go deep with the world s biggest brands and we have fun doing it. Now, we re calling all you rule-breakers and risk-takers who see the world differently, and are bold enough to reinvent it. Come, transform with us.
Are you the one we are looking for We are inviting applications for the role of MT, Publishing
In this role, you will work on ICH guidelines and CTD Structure & Expertise with Publishing Tools like Liquent, Docubridge, etc.
Responsibilities
The Role demands for an expert Publisher with proven ability to execute responsibility in a highly regulated & process driven environment, the Person will be responsible for all the activities related to

    Performing final technical quality review and technical validation (eCTD/NeeS) for US, LATAM and Canadian Submissions.
    Performing final technical quality review and technical validation (eCTD/NeeS)
    Dispatching submission to the meaningful authority (eCTD, NeeS, paper) or affiliate so that affiliate can dispatch to authority;
    Performing post-submission processing activities such as receiving acknowledgement from authority of submission receipt; capturing and the electronic receipt and metadata in RIM; presenting submission receipt to key partners;
    Capturing submissions-related correspondence from health authorities, such as uploading documentation, commitments and metadata.

Qualifications we seek in youMinimum qualifications

    B. Pham/Science Graduate

Preferred qualifications

    Excellent written and verbal communication skills
    Proficient in MS Office applications, especially in MS excel

Genpact is an Equal Opportunity Employer and considers applicants for all positions without regard to race, color, religion or belief, sex, age, national origin, citizenship status, marital status, military/veteran status, genetic information, sexual orientation, gender identity, physical or mental disability or any other characteristic protected by applicable laws. Genpact is committed to creating a dynamic work environment that values diversity and inclusion, respect and integrity, customer focus, and innovation. For more information, visit www.genpact.com. Follow us on Twitter, Facebook, LinkedIn, and YouTube.

,

Other details

    Department:
    Operations Management / Process Analysis
    Industry:
    IT - Software
    Skills:
    accounts,  finance,  accounting,  sales,  reporting,  ms office,  customer focus,  digital conversion,  verbal communication,  go,  rim,  ctd,  ich,  nees,  latam,  visit,  color,  twitter,  dispatch,  facebook 

 
Recruiter details

    Company Name: Genpact
    Company Description:
    GENPACT India Private Limited
 """
extract(text)
text1 = """Customer Support Executive
Invensis Technolgies
Bangalore(JP Nagar )
12th Pass (HSE), Any Graduate
20000 - 25000 Per Month
Customer Care Executive, Inbound Calling, Telecaller, Voice Support
Customer Support Executive Jobs in Bangalore - Invensis Technolgies 
Last Date 31 May 2020
Apply Now

Invensis Technolgies - Job Details
Date of posting: 01 Apr 20
Hi,

Greetings from Invensis Technologies - Bangalore. We are looking for Customer Support Representatives, who can join us immediately.

Candidate should possess Good Communication Skills. 
Manage large amounts of inbound and outbound calls in a timely manner.
Identify customers’ needs, clarify information, research every issue and provide solutions and/or alternatives.
Build sustainable relationships and engage customers by taking the extra mile.
Keep records of all conversations in the call centre database in a comprehensible way.
Meet personal/team qualitative and quantitative targets.
Job Summary

Job Type : Full Time
Job Role : Customer Service / Tech Support
Job Category : Tech Support
Hiring Process : Face to Face Interview, Telephonic Interview
Who can apply : Freshers
About Invensis Technolgies
Invensis Technologies is a 2-decade old company. We are in existence from the past 20 years. We are into BPO, Health Care and Learning. We are a leading IT BPO service provider."""
extract(text1)
text2 = """
Inside Sales Executive
NETMAGIC IT SERVICES PRIVATE LIMITED
Apply
Job Description

    A minimum of 1 - 3 years successful New Business sales experience, incorporating value/service selling.
    B2B Sales experience.
    Goal oriented with superior work ethic.
    Proven territory development skills.
    Experience working in large complex selling environments
    2-3 yea yrs of experience in solution selling / inside sales experience in IT services.
    Proven track record in identifying and pursuing large number of accounts over the phone
    2- 5 years of technology related sales as an inside sales representatives, business development, or sales engineering/consulting experience
    Proven IT sales track record

    Possesses superior follow up skills with the ability to respond under pressure

Other details

    Department:
    Marketing / Communication
    Industry:
    Telecom / ISP
    Skills:
    sales,  cold calling,  marketing,  lead generation 

 
Recruiter details

    Company Name: NETMAGIC IT SERVICES PRIVATE LIMITED
    Company Description:

    About NTT Ltd.

    NTT Ltd. is a leading global technology services company bringing together 28 brands including NTT Communications, Dimension Data, and NTT Security. We partner with organizations around the world to shape and achieve outcomes through intelligent technology solutions. For us, intelligent means data driven, connected, digital, and secure. As a global ICT provider, we employ more than 40,000 people in a diverse and dynamic workplace that spans 57 countries and regions, trades in 73 countries and regions, and delivers services in over 200 countries and regions. Together we enable the connected future. Visit us at our new website www.hello.global.ntt"""
extract(text2)
text3 = """International Human Resource Operations
Hinduja Global Solutions Limited (HGS)
Apply
Job Description
Hindhuja Global Solutions is Hiring for an International HR Operations 
Job Description
 LEVEL: 11 Senior Position
JOB TITLE: Workforce Administration Analyst
(Onboarding, People Movement, Employee & Employment Data Management) 
 Post Employee Onboarding, Global Experience Like UK,USA, UAE , HR Support Role, HR Backend
Education Equivalent 
Any Graduate and above with Min 2yrs to 4 yrs experience in HR Exp
Experience
At least 2 to 4 years of HR Experience
Workforce administration experience is a plus
Workday Experience is a plus
HR Domain certification would be a plus Strong MS Office and Excel skills
LEVEL 11

Max Salary (Monthly CTC)

Bangalore

Workforce administration Associate-L11

29000

Mumbai

Workforce administration Associate-L11

27000
SHIFT IS USA TIME ZONE

LEVEL: 12 Junior Position
JOB TITLE: Workforce Administration Analyst

(Onboarding, People Movement, Employee & Employment Data Management) 
 Post Employee Onboarding, Global Experience Like UK,USA, UAE , HR Support Role, HR Backend
Education Equivalent 

Any Graduate and above with Min 6 months to 2 yrs experience in HR Exp

Experience

At least 6 months to 2 years of HR Experience
Workforce administration experience is a plus
Workday Experience is a plus
HR Domain certification would be a plus Strong MS Office and Excel skills

LEVEL - 12

Max Salary (Monthly CTC)

Bangalore

Workforce administration Associate-L12

27000

Mumbai

Workforce administration Associate-L12

26000

SHIFT IS USA TIME ZONE
INTERVIE PROCESS
HR ROUND
VOICE AND ACCENT
OPS ROUND
VERSANT ROUND 58 CUT OFF
Contact Person- Devika 

Share your resume to devika.hk@hgsbs.com Or Call 9353357764 for more details.

Other details
Department:HRIndustry:IT - SoftwareSkills:english, onboarding, excel, education, max, salary, administration, powerpoint, ms office, hr, analysis, employee onboarding, workday, microsoft word, microsoft excel, windows, teamwork, microsoft outlook, backend, versant
 
Recruiter detailsCompany Name: Hinduja Global Solutions Limited (HGS)Company Description:
Hinduja Global Solutions is an Indian pure play business and service provider headquartered in bangalore, and part of the Hinduja Group. Formerly known as HTMT Global, the company re-branded itself as HGS in line with the group policy.

Email: devika.hk@hgsbs.comTelephone: 9353357764"""
extract(text3)
text4 = """Business Analyst
Brace infotech private limited
Bengaluru / Bangalore5 - 10 Years Not Specified
APPLY
Posted On: 9 hours ago Total Views: 45 Total Applications : 8 Job Id: 24817351Permanent Job
Job Description
Hi All,
This is Mona from Braceinfotech!
Brace Infotech Private Ltd is an emerging Company with one year experience and a an Expert in providing Software Solutions and Talent Acquisition and Recruitment Process Outsourcingsolutions to the Technology, Knowledge Services, Banking and Financial Services, ECommerce, Manufacturing sectors in India, Our Aim is to provide fast, simple and cost-effective solutions for our clients, through strategic advice and solutions in the areas of Recruitment Process Outsourcing, Talent Acquisition Strategy and Planning, Leadership Acquisition, Talent Branding with a Strong and Quick Track Record in the last one year.
We are looking for Business Analyst with minimum 5 – 10 yrs of experience for Bangalore location
Below is the job description:
Business Analyst Senior
MUST Key Skills :
Loan Origination, Loan Servicing experience is a must, Lending Concepts, Collateral etc along with end-end Banking experience.
Mode – Permanent
Location - Bangalore
If you are interested, please share your resume to [HIDDEN TEXT] with below details
Current ctc –
Expected ctc –
Notice period –
Relevant Experience -
Regards,
Mona
Recruitment Consultant

 NOT REGISTERED ON MONSTER YET?
REGISTER NOW
Job Details
Industry:
IT/Computers - Software

Function:
IT

Roles:
Software Engineer/Programmer

Skills:
Business Analyst loan origination loan servicing"""
extract(text4)
text5 = """Account Manager ( East and Bangladesh ) in Remote | Sales at citrix
Citrix Systems India Pvt Ltd
card_travel12 to 15 yrs₹As per Industry Standardslocation_onBangladesh (Bangladesh)
!
Posted on 22 Apr, 2020
JOB DESCRIPTION
We believe work is not a place, but rather a thing you do. Our technology revolves around this core philosophy. We are relentlessly committed to helping people work and play from anywhere, on any device. Innovation, creativity and a passion for ever-improving performance drive our company and our people forward. We empower the original mobile device: YOU!


What we're looking for:


Position Summary

The account manager will be responsible for managing Citrix business with companys most strategic & important named accounts in the East Region of India and Bangladesh. Typically manages a mix of up to 75 enterprise and corporate accounts across all vertical segments of markets. The role also involves orchestrating Citrix GTM in the region by actively partnering with extended teams in Citrix namely channels, marketing, presales, etc.


Primary Duties / Responsibilities

Establish relationships and engage at executive levels within assigned accounts to identify and sell Citrix products and services direct to the companys most strategic, complex named accounts. Strong customer interaction at the Sr. Management level with support on forming a relationship with C level contacts within these accounts.
Provides professional leadership, coordination and direction to the team.
Build up strategic contacts at CIO, managing director and executive level
Carry a revenue quota to meet or exceed sales targets and demonstrate continuous progress towards achieving account strategies.
Draw up and implement analysis of key accounts, using Citrix methodology
Identify, develop, execute, and maintain account strategies to drive adoption of Citrix product and services revenue within assigned enterprise accounts and their subsidiaries and affiliates. Accounts are key, strategic and may have complex requirements.
Establish and lead teams of internal and external resources to identify, pursue, and close specific opportunities consistent with account strategies.
Take on responsibility for overall project management from acquisition of new IT projects to coordination of project participants.
Establish and maintain close relationships with inside sales, systems engineers, consultants, and sales specialists to access to and leverage from appropriate internal resources.
Establish and maintain relationships with resellers, system integrators, and any other external partner to develop and achieve account strategies and opportunity plans.
Understand and navigate account procurement practices to successfully negotiate profitable licensing contracts.
Drive prompt resolution of customer issues and ensure high levels of customer satisfaction with Citrix products and services.
Serve as the primary client contact for non-technical or support issues requiring escalation.
Provide regular and efficient updates on assigned accounts to sales management.
Ensure accurate and timely forecasts in CRM.
Should have a strong Hunting Experience


Qualifications (knowledge, skills, abilities)

Demonstrated track record of establishing strategic executive level relationships to position and sell software and services to listed companies.
Proven ability to develop and maintain executive level relationships with both business and IT customers.
Proven ability to develop and maintain strong working relationships with internal marketing, technology, implementation, and product development teams.
Strong consultative selling ability critical questioning, listening, analytical, negotiation, communication, and presentation.
Demonstrated ability to develop and articulate compelling qualitative and quantitative business cases for IT solutions.
Proven ability to manage long, complex sales cycles from beginning to end and ability to close large complex deals with enterprise accounts.
Demonstrated knowledge of strategic/large account sales techniques and processes including the ability to understand customer needs, overcome objections, develop business cases, and negotiate and close deals.
High energy, motivated self-starter attitude
Excellent written and verbal communication skills including the ability to effectively present to both technical and executive audiences in the local language and in English.


Requirements (Education, Certification, Training, and Experience)

Bachelors degree or equivalent experience required
12 years of sales experience, with at least 5 years selling software/hardware to large enterprise accounts and strong history of quota achievement.
Viewed as an expert in application software or networking sales in the company and/or industry.
Track record of meeting or exceeding enterprise sales quotas
Proven track record in closing complex enterprise sales.
Experience working with external partners to develop and close business within enterprise accounts.
Experience in negotiating complex licensing agreements.
Network of enterprise reference contacts with whom one has built a trust relationship and with whom one can gain an audience to present Citrix solutions and identify opportunities.
Experience in a minimum of 1 to 2 of the following areas of expertise: (i) application and core technology (database, operating systems, application platforms, hardware, networking); (ii) Internet solution selling (service, hardware, software); and/or (iii) networking, security and/or WAN optimization.


Physical Demands / Work Environment

Ability to travel as appropriate in order to manage and close opportunities within assigned accounts and their subsidiaries and affiliates.

Work with minimal supervision.

Person must be flexible and able to deal with fast and rapid change.


What youre looking for: Our technology is built on the idea that everyone should be able to work from anywhere, at any time, and on any device. Its a simple philosophy that guides everything we do including how we work. If youre in sales, well help you make your numbers and a difference with a brand you can believe in. We want employees to do what they do best, every day.


Be bold. Take risks. Imagine a better way to work. If we just described you, then we really need to talk.


Functional Area: Territory Rep (Field Sales Manager)

About us:

Citrix is a cloud company that enables mobile workstyles. We create a continuum between work and life by allowing people to work whenever, wherever, and however they choose. Flexibility and collaboration is what were all about. The Perks: We offer competitive compensation and a comprehensive benefits package. Youll enjoy our workstyle within an incredible culture. Well give you all the tools you need to succeed so you can grow and develop with us.


Citrix Systems, Inc. is firmly committed to Equal Employment Opportunity (EEO) and to compliance with all federal, state and local laws that prohibit employment discrimination on the basis of age, race, color, gender, sexual orientation, gender identity, ethnicity, national origin, citizenship, religion, genetic carrier status, disability, pregnancy, childbirth or related medical conditions, marital status, protected veteran status and other protected classifications.


Citrix uses applicant information consistent with the Citrix Recruitment Policy Notice at https://www.citrix.com/about/legal/privacy/citrix-recruitment-privacy-notice.html


Citrix welcomes and encourages applications from people with disabilities. Reasonable accommodations are available on request for candidates taking part in all aspects of the selection process. If you are an individual with a disability and require a reasonable accommodation to complete any part of the job application process, please contact us at (877) 924-8749 or email us at ASKHR@citrix.com for assistance.


If this is an evergreen requisition, by applying you are giving Citrix consent to be considered for future openings of other roles of similar qualifications.
JOB FUNCTION:Accounting / Tax / Company Secretary / Audit
INDUSTRY:Software Services
SPECIALIZATION:Other Accounting
QUALIFICATION:
Any Graduate
KEY SKILLS
consultative sellingenterprise salespresentationsolution sellingnegotiationgtmarticulatetargetsaccount salescustomer interactionsales managementkey accountscorporate accountsinnovationinside salesmanagementbuild
JOB POSTED BY
COMPANY:Citrix Systems India Pvt Ltd
WEBSITEhttp://www.citrix.com
INDUSTRYTechnology (IT, Telecom, Dot Com etc) (Managed IT Services (MSPs))
COMPANY TURNOVER10000 - 10000+ Crores
COMPANY SIZE5001 - 10000 Employees"""
extract(text5)