import  re

import pandas as pd 

import os



def  getPaperDetails():

    #reading the sentences in all the documents

    paperDict = dict()

    paperName = dict()

    readFile2 = r'/kaggle/input/kernel2ee16e573a/Extracted_sentences_from_filtered_covid_documents.csv'

    dfP = pd.read_csv(readFile2, sep='\t' )

    for row in dfP.itertuples(): 

        paper = row.Cord_uid

        Name = row.Titles

        paperName[paper] = Name

        if paper in paperDict.keys():

            sentList = paperDict[paper]

            sentList.append(row.Sentence)

            paperDict[paper]= sentList

        else :

            sentList = []

            sentList.append(row.Sentence)

            paperDict[paper]= sentList

            

    print("Total papers read : ", len(paperDict.keys()))

    return paperDict, paperName



print('reading document details')

#print(os.listdir("../input"))

paperDict, paperName = getPaperDetails()

sampleDict = dict()

sampleMethodDict = dict()

designDict = dict()

sevDict = dict()

fatalDict = dict()

dfObj = pd.DataFrame(columns=['Cord_uid', 'Titles', 'Severe', 'Fatal', 'Design', 'Sample', 'Sampling Method'])



#for displaying documents where information was detected

dfObj2 = pd.DataFrame(columns=['Cord_uid', 'Titles', 'Severe', 'Fatal', 'Design', 'Sample', 'Sampling Method'])



searchVal = ['P=', 'P<', 'P>', 'P =', 'P <', 'P >', 'p=', 'p<', 'p>', 'p =', 'p <', 'p >', 'p-value']



#regex OR [0-9.]* \([ 0-9%A-Z:.\-]+\)

wb = ['OR', 'AOR', 'HR', 'AHR', 'OR of', 'HR of', 'HR was', 'OR was', 'odds ratio', 'Odds Ratio', 'odds ratio of', 'Odds Ratio of','odds ratio was', 'Odds Ratio was']





tags = dict()

tags['smoke'] =  'smoke smoking'

tags['pulmonary'] = 'pulmonary lung COPD bronchitis emphysema'

tags['respiratory'] = 'respiratory breathe wheeze'

tags['diabetes'] = 'diabetes insulin diabetic sugar glycemia glucose'

tags['asthma'] = 'asthma'

tags['comorbidity'] = 'comorbidit co-morbidit'

tags['pregnant'] = 'neonat pregnan born postpartum post-partum baby mother'

tags['hypertension'] = 'hypertension'

tags['cerebral'] = 'cerebral brain'

tags['cancer'] = 'cancer tumour tumor carcino'

tags['obesity'] = 'obesity'

tags['heart'] = 'cardiac cardio heart stroke'

tags['alcohol'] = 'drink alcohol drunk'

tags['tuberculosis'] = 'tuberculosis'

tags['kidney'] = 'kidney'

tags['liver'] = 'liver'

tags['reproduction']  = 'reproduction'

for i in paperDict.keys():

    sentList = paperDict[i]

    matchList = []

    dsl = []

    sD = []

    fs = []

    test_list = ['.we ', '.our ', '.in this ', '.here ', ' we ', ' here ', ' our ', ' in this ', 'this']

    ignoreList = ['et al', 'colleagues']

    for sent in sentList:

        ic = any(ele in sent.lower() for ele in ignoreList)

        if ic:

            continue

            

        #sample extraction

        match = re.search(r'[\d,]+'+' patients', sent)

        if match and ('study' in sent.lower() or 'sample' in sent.lower() or 'enroll' in sent.lower()):

            mg = match.group()

            if re.search(r'[\d]+', mg):

                sampleDict[i] = match.group()

        

        se = sent

        if (' Hospital' in se or ' Clinic ' in se or ' Database' in se or ' Clinical ' in se) and ('study' in sent or 'collect' in se.lower() or 'conduct' in se.lower() or 'enroll' in se.lower()  or 'approve' in se.lower()) and ('result' not in se or 'found' not in se.lower()):

            sampleMethodDict[i] = se

                

        res = any(ele in sent.lower() for ele in test_list) 

        #extracting design type

        if ('retrospective' in sent.lower()) and  (('study' in sent.lower()) or ('analysis' in sent.lower()) or ('cohort' in sent.lower())):

            if res:

                #print('found here', sent)

                dsl.append('Retrospective Study')

                #print(dsl)

                

        if 'prospective' in sent.lower() and  ('study' in sent.lower() or 'analysis' in sent.lower() or 'cohort' in sent.lower()):

            if res:

                dsl.append('Prospective Study')

        

        if ('cross-section' in sent.lower() or 'cross section' in sent.lower()) and  ('case' in sent.lower() or 'control' in sent.lower()):

            if res:

                dsl.append('Cross-sectional Case-control')

            

        if ('match' in sent.lower() and  ('case' in sent.lower() or 'control' in sent.lower())):

            if res:

                dsl.append('Matched Case-Control')

        

        if 'prevalence' in sent.lower() and 'prevalence of' not  in  sent.lower() and  ('survey' in sent.lower() or 'surveillance' in sent.lower()):

            if res:

                dsl.append('Prevalence Survey Study')

            

        if 'time' in sent.lower() and  'series' in sent.lower() and ( 'event' in sent.lower() or 'analysis' in sent.lower()):

            if res:

                dsl.append('Time Series Analysis')

            

        if ('systematic' in sent.lower() or 'meta ' in sent.lower() or 'meta-' in sent.lower()) and 'meta-data' not in sent.lower() and  ('analysis' in sent.lower() or 'review' in sent.lower()):

            if res:

                dsl.append('Systematic review and meta-analysis')

            

        if 'random' in sent.lower() and ('control ' in sent.lower() ):

            if 'pseudo ' in sent.lower() or 'quasi' in sent.lower() or 'non-random' in sent.lower() or 'non random' in sent.lower() or 'nonrandom' in sent.lower():

                

                if res:

                    dsl.append('Pseudo randomized controlled study')

            else: 

                if res:

                    dsl.append('Randomized controlled study')

                    

                    

        #severe fatal

        for opt in wb:

            

            match = re.findall(opt+r'[.;=0-9, ]* [\(\[0-9% A-z.:\-;=\)\]]+', sent)

            

            if match:

                #print(match)

                ty = 'severe'

                if 'death' in sent.lower() or 'fatal' in sent.lower() or 'mortal' in sent.lower():

                    ty = 'fatal'

                for t in tags.keys():

                    ts = tags[t].split(' ')

                    for tg in ts:

                        if tg in sent.lower():

                            #print(">>>>>>>> ",sent)

                            ss = sent.split(tg)

                            if len(ss) > 1:

                                m = re.search(opt+r'[.;=0-9, ]* [\(\[0-9% A-z.:\-;=\)\]]+', ss[1])

                                

                            else:

                                m = re.search(opt+r'[.;=0-9, ]* [\(\[0-9% A-z.:\-;=\)\]]+', ss[0])

                            

                            if m:

                                

                                st = t+' : '+m.group()

                                st=st.strip('.')

                                m2 = re.search(r'[\d]+', st)

                                

                                if '(' not in st:

                                    st = st.replace(')', '')

                                if m2:

                                    if not any(ele in st.lower() for ele in ['a', 'the', 'in', 'at']):

                                        if ty == 'severe':

                                            sD.append(st)

                                        else:

                                            fs.append(st)

                                    

                                    #print(st)



        for sv in searchVal:

            match = re.findall(sv+r'[0-9. ]+', sent)

            if match:

                #print(match)

                ty = 'severe'

                if 'death' in sent.lower() or 'fatal' in sent.lower() or 'mortal' in sent.lower():

                    ty = 'fatal'

                for t in tags.keys():

                    ts = tags[t].split(' ')

                    for tg in ts:

                        if tg in sent.lower():

                            ss = sent.split(tg)

                            if len(ss) > 1:

                                m = re.search(sv+r'[0-9. ]+', ss[1])  

                            else:

                                m = re.search(sv+r'[0-9. ]+', ss[0])

                            if m:

                                st = t+' : '+m.group()

                                if '(' not in st:

                                    st = st.replace(')', '')

                                m2 = re.search(r'[\d]+', st)

                                if m2:

                                    if not any(ele in st.lower() for ele in ['a', 'the', 'in', 'at']):

                                        if ty == 'severe':

                                                sD.append(st)

                                        else:

                                            fs.append(st)

                                    

                                    #print(st)

                                

    sevDict[i] = list(set(sD))

    fatalDict[i] = list(set(fs))

                    

    if len(dsl)>0:

        designDict[i] = max(set(dsl), key = dsl.count)

    else :

        designDict[i] = ''

    

    #designDict[i] = dsl

    if i not in sampleDict.keys():

        sampleDict[i] = ''

    if i not in sampleMethodDict.keys():

        sampleMethodDict[i] = ''

        

    #writing to dataframe

    dfObj = dfObj.append({'Cord_uid':i, 'Titles': paperName[i], 'Severe':sevDict[i], 'Fatal':fatalDict[i], 'Design': designDict[i], 'Sample': sampleDict[i], 'Sampling Method':sampleMethodDict[i]},ignore_index=True)

    

    #documents where value was found

    if designDict[i] != '' or sampleDict[i] != '' or sampleMethodDict[i] != '' or sevDict[i]!= []  or fatalDict[i]!= []:

        dfObj2 = dfObj2.append({'Cord_uid':i, 'Titles': paperName[i], 'Severe':sevDict[i], 'Fatal':fatalDict[i], 'Design': designDict[i], 'Sample': sampleDict[i], 'Sampling Method':sampleMethodDict[i]},ignore_index=True)

    

from IPython.display import display, HTML

display(HTML(dfObj2[:15].to_html()))

print('Saving result for all documents in extracted_DocData.csv ...')

dfObj.to_csv('extracted_DocData.csv', sep = '\t')

print('Saved')