# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
df_detail = pd.read_csv("/kaggle/input/complaints-data/complaints-accused_2000-2016_2016-11.csv")
df_victims = pd.read_csv("/kaggle/input/complaints-data/complaints-victims_2000-2016_2016-11.csv")
df_main = pd.read_csv("/kaggle/input/complaints-data/complaints-complaints_2000-2016_2016-11.csv")
df_main.head()
df_main.columns
#remove unnecessary info
df_main = df_main.drop(["row_id", "location_code", "address_number", "street", "apartment_number", "incident_time", "complaint_date","closed_date"], axis=1)
#creating year and dropping date columns
df_main['incident_year'] = df_main['incident_date'].apply(lambda x: int(x[0:4]))
df_main.drop(['incident_date'], axis = 1, inplace = True)
#pruning our data to only contain CR_ID's >= 1000000 as these are the CR_ID's present across all data sets
df_main = df_main[df_main.cr_id >= 1000000]
df_main.head()
#open and read CPD beats file
beatset = set([])
beatf = open("/kaggle/input/complaints-data/beatlist.txt",'r')
N = int(beatf.readline()[:-1])
for i in range(N):
    beatset.add(beatf.readline()[:-1])
#helper func to easily pull beat names and ensure that each beat is a part of the beats file we previously opened and read
def pad0(string, length):
    if (type(string) != str):
        string = str(int(string))
    ans = ""
    for i in range(length-len(string)):
        ans += '0'
    ans += string
    if (ans in beatset):
        return ans
    else:
        return "asdf"

#helper func to ensure all complaints in the Chicago-Metro Area
def city_state_checker(string):
    if string[:2] != "CH" or string == "CHICAGO RIDG IL":
        return "bAd"
    return "gOOd"
#applying above funcs to data for removing data with irrelevant beats and formatting exisiting beats
df_main = df_main[~pd.isna(df_main.beat)]
df_main.beat = df_main.beat.apply(lambda x: pad0(x, 4))
df_main = df_main[df_main.beat != "asdf"]
#applying above helper funcs to remove all complaints not in our geographic area of interests
df_main = df_main[~pd.isna(df_main.city_state)]
df_main.city_state = df_main.city_state.apply(lambda x: city_state_checker(x))
df_main = df_main[df_main.city_state != "bAd"]
#the above function removed need for 'city_state' column so let's drop
df_main = df_main.drop(["city_state"], axis=1)
#index by CR ID for easy sorting/combining
df_main = df_main.set_index("cr_id")
df_main.head()
df_victims.head()
#remove null values for race
df_victims = df_victims[~pd.isna(df_victims["race"])]
#set index to CR ID for easy sorting/combining
df_victims = df_victims.set_index("cr_id")
df_victims.head()
df_detail.head()
#removing all CR ID < 1000000
df_detail = df_detail[df_detail["cr_id"] >= 1000000]
#set index to CR ID for easy sorting/combining
df_detail = df_detail.set_index("cr_id")
#drop an unnecessary column
df_detail = df_detail.drop(["complaints-accused_2000-2016_2016-11_ID", "UID"], axis=1)
df_detail.head()
df_main = df_main.sort_index()
df_victims = df_victims.sort_index()
df_detail = df_detail.sort_index()
df_main.drop_duplicates(inplace = True)
df_detail.drop_duplicates(inplace = True)
df_main.head()
df_victims.head()
df_detail.head()
df_main.shape
df_victims.shape
df_detail.shape
df_main.head()
#graph for beats with most complaints
beatFreq = df_main.beat.value_counts()[:20]
beatFreq_beats = beatFreq.index
sns.set(font_scale = 1)
ax = sns.barplot(x = beatFreq_beats, y = beatFreq, order = beatFreq_beats, label = 'small')
ax.set_title('20 Beats with HIGHEST Number of Complaint Reports')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set(xlabel='Beats', ylabel='Number of Complaints')
ax.tick_params(labelsize=8.5)
plt.show()
#graph for beats with least complaints
beatFreq_low = df_main.beat.value_counts()[-20:]
beatFreq_low_beats = beatFreq_low.index
sns.set(font_scale = 1)
ax = sns.barplot(x = beatFreq_low_beats, y = beatFreq_low, order = beatFreq_low_beats, label = 'small')
ax.set_title('20 Beats with LOWEST Number of Complaint Reports')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set(xlabel='Beats', ylabel='Number of Complaints')
ax.tick_params(labelsize=8.5)
plt.show()
#counter dict for district with most beats in top 20
districts_high = {}
for beat in beatFreq_beats:
    district = beat[:2]
    if district in districts_high.keys():
        districts_high[district] += 1
    else:
        districts_high[district] = 1
#print(districts_high)
districts_high_sorted_keys = sorted(districts_high, key=districts_high.get, reverse=True)
districts_high_sort ={}
for district in districts_high_sorted_keys:
    districts_high_sort[district] = districts_high[district]
print(districts_high_sort)
#graph of above counter
sns.set(font_scale = 1)
ax = sns.barplot(x = list(districts_high_sort.keys()), y = list(districts_high_sort.values()), order = districts_high_sort, label = 'small')
ax.set_title('Districts with HIGHEST Number of Complaints')
ax.set_xticklabels(ax.get_xticklabels())
ax.tick_params(labelsize=8.5)
ax.set(xlabel='Districts', ylabel='Number of Complaints')
plt.show()
#counter dict for district with most beats in bottom 20
districts_low = {}
for beat in beatFreq_low_beats:
    district = beat[:2]
    if district in districts_low.keys():
        districts_low[district] += 1
    else:
        districts_low[district] = 1
#print(districts_low)
districts_low_sorted_keys = sorted(districts_low, key=districts_low.get, reverse=True)
districts_low_sort ={}
for district in districts_low_sorted_keys:
    districts_low_sort[district] = districts_low[district]
print(districts_low_sort)
#graph of above counter
sns.set(font_scale = 1)
ax = sns.barplot(x = list(districts_low_sort.keys()), y = list(districts_low_sort.values()), order = districts_low_sort, label = 'small')
ax.set_title('Districts with LOWEST Number of Complaints')
ax.set_xticklabels(ax.get_xticklabels())
ax.tick_params(labelsize=8.5)
ax.set(xlabel='Districts', ylabel='Number of Complaints')
plt.show()
print(df_main.incident_year.value_counts())
#graph of frequency each year appears across all available complaint data
ax = sns.countplot(x="incident_year", data=df_main)
ax.set_title('Number of Complaints Filed by year')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()
df_victims.head()
#counter for gender frequency across all available complaints
print(df_victims.gender.value_counts())
#graph of above counter
ax = sns.countplot(x="gender", data=df_victims,  order = df_victims.gender.value_counts().index)
ax.set_title('Breakdown of Complainees by Gender')
plt.show()
#counter for race frequency across all available complaints
print(df_victims.race.value_counts())
#graph of above counter
ax = sns.countplot(x="race", data=df_victims, order = df_victims.race.value_counts().index)
ax.set_title('Breakdown of Complainees by Race')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()
#helper func to create age ranges
def bander(age):
    if (age<20):
        return '<20'
    elif(age>=20 and age<30):
        return '20-29'
    elif(age>=30 and age<40):
        return '30-39'
    elif(age>=40 and age<50):
        return '40-49'
    elif(age>=50 and age<60):
        return '50-59'
    elif(age>=60 and age<70):
        return '60-69'
    else:
        return '>=70'
#apply helper func to complete complaints data
df_victims['age_band'] = df_victims.age.apply(lambda x: bander(x))
#view of frequency of age band
df_victims.age_band.value_counts()
#graph of above counter
age_band_order = ['<20', '20-29','30-39','40-49','50-59','60-69', '>=70' ]
ax = sns.countplot(x="age_band", data=df_victims, order = age_band_order)
ax.set_title('Breakdown of Complainees by Age')
plt.show()
df_victims.drop('age', axis = 1, inplace = True)
pd.set_option('display.max_rows', None)
table = pd.pivot_table(df_victims, index=['race', 'gender', 'age_band'], aggfunc= len, fill_value = 0)
print(table)
df_detail.head()
#store all the complaint categories in a neat list
complaint_cat_list = list(df_detail.complaint_category.unique())
#create a dict for the above said 3-character key and the corresponding description
complaint_cat_dict = {}
for x in complaint_cat_list:
    desc = str(x)
    key = desc[:3]
    DESC = desc[4:]
    complaint_cat_dict[key] = DESC
print(len(complaint_cat_dict))
#remove the descriptions from the complaints
df_detail['complaint_key'] = df_detail.complaint_category.apply(lambda x: str(x)[:3])
#we don't need all full descriptions anymore, so drop
df_detail.drop('complaint_category', axis = 1, inplace = True)
#replacing null values with 'OTHER'
df_detail['complaint_key'].replace('nan', 'OTHER', inplace = True)
#before removing dups (uncomment if needed but long)
#df_detail.complaint_key.value_counts()
#removing some auxilary category
df_detail.drop('row_id', axis = 1, inplace = True)
#creating our no dups dataframe
df_detail_no_dups = df_detail.drop_duplicates()
#after removing dups (uncomment if needed but long)
#df_detail_no_dups.complaint_key.value_counts()
#table of 20 most frequent complaint types in duplicates data
print('20 Most Frequent Complaint Types (With Duplicates) — Table\n')
for key in df_detail.complaint_key.value_counts()[:20].index:
    if(key == 'OTHER'):
        print('OTHER')
        print("Frequency: "+str(df_detail.complaint_key.value_counts()[key]))
        print('-'*50)
    else:
        print(key+": "+complaint_cat_dict[key])
        print("Frequency: "+str(df_detail.complaint_key.value_counts()[key]))
        print('-'*50)
#graph of above data via frequency
sns.set(font_scale = 1)
ax = sns.barplot(x = df_detail.complaint_key.value_counts()[:20].index, y = df_detail.complaint_key.value_counts()[:20], order = df_detail.complaint_key.value_counts()[:20].index, label = 'small')
ax.set_title('20 Most Frequent Complaint Types (With Duplicates)')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set(xlabel='Complaint Type', ylabel='Frequency')
ax.tick_params(labelsize=8.5)
plt.show()
#table of 20 most frequent complaint types in no duplicates data
print('20 Most Frequent Complaint Types (No Duplicates) — Table\n')
for key in df_detail_no_dups.complaint_key.value_counts()[:20].index:
    print(key+": "+complaint_cat_dict[key])
    print("Frequency: "+str(df_detail_no_dups.complaint_key.value_counts()[key]))
    print('-'*50)
#graph of above data via frequency
sns.set(font_scale = 1)
ax = sns.barplot(x = df_detail_no_dups.complaint_key.value_counts()[:20].index, y = df_detail_no_dups.complaint_key.value_counts()[:20], order = df_detail_no_dups.complaint_key.value_counts()[:20].index, label = 'small')
ax.set_title('20 Most Frequent Complaint Types (No Duplicates)')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set(xlabel='Complaint Type', ylabel='Frequency')
ax.tick_params(labelsize=8.5)
plt.show()
df_detail.reset_index(level = 0, inplace = True)
df_detail_no_dups.reset_index(level = 0, inplace = True)
#discipline code counter for duplicate data + sorting (recommended discipline)
discipline_dict_rec = {}
discipline_dict_rec['x Day Suspension'] = 0
discipline_dict_rec['Over 30 Day Suspension'] = 0
discipline_dict_rec['Reprimanded'] = 0
discipline_dict_rec['Administrative Termination'] = 0
discipline_dict_rec['Year-long Suspension or Longer'] = 0
discipline_dict_rec['Seperation'] = 0
discipline_dict_rec['Reinstated'] = 0
discipline_dict_rec['Nothing'] = 0
discipline_dict_rec['Resigned'] = 0
for ind in df_detail.index:
    rec = df_detail['recommended_discipline'][ind]
    if rec <199:
        discipline_dict_rec['x Day Suspension'] += 1
    if rec == 200:
        discipline_dict_rec['Over 30 Day Suspension'] +=1
    if rec == 0 or rec == 100:
        discipline_dict_rec['Reprimanded'] += 1
    if rec == 300:
        discipline_dict_rec['Administrative Termination'] +=1
    if rec == 365:
        discipline_dict_rec['Year-long Suspension or Longer'] +=1
    if rec == 400:
        discipline_dict_rec['Seperation'] += 1
    if rec == 500:
        discipline_dict_rec['Reinstated'] += 1
    if rec == 600 or rec == 900:
        discipline_dict_rec['Nothing'] += 1
    if rec == 800:
        discipline_dict_rec['Resigned'] += 1
discipline_dict_rec_keys = sorted(discipline_dict_rec, key=discipline_dict_rec.get, reverse=True)
discipline_dict_rec_sort ={}
for district in discipline_dict_rec_keys:
    discipline_dict_rec_sort[district] = discipline_dict_rec[district]
print(discipline_dict_rec_sort)
#discipline code counter for duplicate data + sorting (final discipline)
discipline_dict_fin = {}
discipline_dict_fin['x Day Suspension'] = 0
discipline_dict_fin['Over 30 Day Suspension'] = 0
discipline_dict_fin['Reprimanded'] = 0
discipline_dict_fin['Administrative Termination'] = 0
discipline_dict_fin['Year-long Suspension or Longer'] = 0
discipline_dict_fin['Seperation'] = 0
discipline_dict_fin['Reinstated'] = 0
discipline_dict_fin['Nothing'] = 0
discipline_dict_fin['Resigned'] = 0
for ind in df_detail.index:
    fin = df_detail['final_discipline'][ind]
    if fin <199:
        discipline_dict_fin['x Day Suspension'] += 1
    if fin == 200:
        discipline_dict_fin['Over 30 Day Suspension'] +=1
    if fin == 0 or fin == 100:
        discipline_dict_fin['Reprimanded'] += 1
    if fin == 300:
        discipline_dict_fin['Administrative Termination'] +=1
    if fin == 365:
        discipline_dict_fin['Year-long Suspension or Longer'] +=1
    if fin == 400:
        discipline_dict_fin['Seperation'] += 1
    if fin == 500:
        discipline_dict_fin['Reinstated'] += 1
    if fin == 600 or fin == 900:
        discipline_dict_fin['Nothing'] += 1
    if fin == 800:
        discipline_dict_fin['Resigned'] += 1
discipline_dict_fin_keys = sorted(discipline_dict_fin, key=discipline_dict_fin.get, reverse=True)
discipline_dict_fin_sort ={}
for district in discipline_dict_fin_keys:
    discipline_dict_fin_sort[district] = discipline_dict_fin[district]
print(discipline_dict_fin_sort)
#putting both counters into an output with some light analysis 
print('DISCIPLINE DATA (with duplictes)\n')
for disc in discipline_dict_rec_sort:
    print(disc)
    print('Frequency that '+disc+' was the Recommended Discipline: '+str(discipline_dict_rec[disc]))
    print('Frequency that '+disc+' was the Final Discipline: '+str(discipline_dict_fin[disc]))
    if discipline_dict_rec[disc]>discipline_dict_fin[disc]:
        print('Difference between Recommended and Final Frequencies: '+str(discipline_dict_rec[disc]-discipline_dict_fin[disc])+' ('+ str(abs(discipline_dict_rec[disc]-discipline_dict_fin[disc]))+' more Recommended than Final)')
    elif discipline_dict_rec[disc]<discipline_dict_fin[disc]:
        print('Difference between Recommended and Final Frequencies: '+str(discipline_dict_rec[disc]-discipline_dict_fin[disc])+' ('+ str(abs(discipline_dict_rec[disc]-discipline_dict_fin[disc]))+' more Final than Recommended)')
    else:
        print('Frequencies of Recommended and Final are Equal')
    print('-'*50)
#discipline code counter for non-duplicate data + sorting (recommended discipline)
discipline_dict_no_dups_rec = {}
discipline_dict_no_dups_rec['x Day Suspension'] = 0
discipline_dict_no_dups_rec['Over 30 Day Suspension'] = 0
discipline_dict_no_dups_rec['Reprimanded'] = 0
discipline_dict_no_dups_rec['Administrative Termination'] = 0
discipline_dict_no_dups_rec['Year-long Suspension or Longer'] = 0
discipline_dict_no_dups_rec['Seperation'] = 0
discipline_dict_no_dups_rec['Reinstated'] = 0
discipline_dict_no_dups_rec['Nothing'] = 0
discipline_dict_no_dups_rec['Resigned'] = 0
for ind in df_detail_no_dups.index:
    rec = df_detail['recommended_discipline'][ind]
    if rec <199:
        discipline_dict_no_dups_rec['x Day Suspension'] += 1
    if rec == 200:
        discipline_dict_no_dups_rec['Over 30 Day Suspension'] +=1
    if rec == 0 or rec == 100:
        discipline_dict_no_dups_rec['Reprimanded'] += 1
    if rec == 300:
        discipline_dict_no_dups_rec['Administrative Termination'] +=1
    if rec == 365:
        discipline_dict_no_dups_rec['Year-long Suspension or Longer'] +=1
    if rec == 400:
        discipline_dict_no_dups_rec['Seperation'] += 1
    if rec == 500:
        discipline_dict_no_dups_rec['Reinstated'] += 1
    if rec == 600 or rec == 900:
        discipline_dict_no_dups_rec['Nothing'] += 1
    if rec == 800:
        discipline_dict_no_dups_rec['Resigned'] += 1
discipline_dict_no_dups_rec_keys = sorted(discipline_dict_no_dups_rec, key=discipline_dict_no_dups_rec.get, reverse=True)
discipline_dict_no_dups_rec_sort ={}
for district in discipline_dict_no_dups_rec_keys:
    discipline_dict_no_dups_rec_sort[district] = discipline_dict_no_dups_rec[district]
print(discipline_dict_no_dups_rec_sort)
#discipline code counter for non-duplicate data + sorting (final discipline)
discipline_dict_no_dups_fin = {}
discipline_dict_no_dups_fin['x Day Suspension'] = 0
discipline_dict_no_dups_fin['Over 30 Day Suspension'] = 0
discipline_dict_no_dups_fin['Reprimanded'] = 0
discipline_dict_no_dups_fin['Administrative Termination'] = 0
discipline_dict_no_dups_fin['Year-long Suspension or Longer'] = 0
discipline_dict_no_dups_fin['Seperation'] = 0
discipline_dict_no_dups_fin['Reinstated'] = 0
discipline_dict_no_dups_fin['Nothing'] = 0
discipline_dict_no_dups_fin['Resigned'] = 0
for ind in df_detail_no_dups.index:
    fin = df_detail['final_discipline'][ind]
    if fin <199:
        discipline_dict_no_dups_fin['x Day Suspension'] += 1
    if fin == 200:
        discipline_dict_no_dups_fin['Over 30 Day Suspension'] +=1
    if fin == 0 or fin == 100:
        discipline_dict_no_dups_fin['Reprimanded'] += 1
    if fin == 300:
        discipline_dict_no_dups_fin['Administrative Termination'] +=1
    if fin == 365:
        discipline_dict_no_dups_fin['Year-long Suspension or Longer'] +=1
    if fin == 400:
        discipline_dict_no_dups_fin['Seperation'] += 1
    if fin == 500:
        discipline_dict_no_dups_fin['Reinstated'] += 1
    if fin == 600 or fin == 900:
        discipline_dict_no_dups_fin['Nothing'] += 1
    if fin == 800:
        discipline_dict_no_dups_fin['Resigned'] += 1
discipline_dict_no_dups_fin_keys = sorted(discipline_dict_no_dups_fin, key=discipline_dict_no_dups_fin.get, reverse=True)
discipline_dict_no_dups_fin_sort ={}
for district in discipline_dict_no_dups_fin_keys:
    discipline_dict_no_dups_fin_sort[district] = discipline_dict_no_dups_fin[district]
print(discipline_dict_no_dups_fin_sort)
#putting both counters into an output with some light analysis 
print('DISCIPLINE DATA (no duplictes)\n')
for disc in discipline_dict_no_dups_rec_sort:
    print(disc)
    print('Frequency that '+disc+' was the Recommended Discipline: '+str(discipline_dict_no_dups_rec[disc]))
    print('Frequency that '+disc+' was the Final Discipline: '+str(discipline_dict_no_dups_fin[disc]))
    if discipline_dict_no_dups_rec[disc]>discipline_dict_no_dups_fin[disc]:
        print('Difference between Recommended and Final Frequencies: '+str(discipline_dict_no_dups_rec[disc]-discipline_dict_no_dups_fin[disc])+' ('+ str(abs(discipline_dict_no_dups_rec[disc]-discipline_dict_no_dups_fin[disc]))+' more Recommended than Final)')
    elif discipline_dict_no_dups_rec[disc]<discipline_dict_no_dups_fin[disc]:
        print('Difference between Recommended and Final Frequencies: '+str(discipline_dict_no_dups_rec[disc]-discipline_dict_no_dups_fin[disc])+' ('+ str(abs(discipline_dict_no_dups_rec[disc]-discipline_dict_no_dups_fin[disc]))+' more Final than Recommended)')
    else:
        print('Frequencies of Recommended and Final are Equal')
    print('-'*50)
#helper func to determine whether the final discipline is more/less/equally severe as recommended
def compare(rec, fin):
    if pd.isnull(rec) or pd.isnull(fin):
        return 'NO DATA'
    if rec == fin:
        return "EQUAL"
    if rec<fin and fin != 600 and fin!= 900 and fin!= 500:
        return "GREATER THAN RECOMMENDED"
    else: 
        return "LESS THAN RECOMMENDED"
#applying helper to both data frames
df_detail['compare_discipline'] = df_detail.apply(lambda row: compare(row['recommended_discipline'], row['final_discipline']), axis = 1)
df_detail_no_dups['compare_discipline'] = df_detail_no_dups.apply(lambda row: compare(row['recommended_discipline'], row['final_discipline']), axis = 1)
#printing counter of comparison func in both data sets (as labeled)
print('Compare Disciplines for DF_DETAIL (with duplicates)\n')
print(df_detail.compare_discipline.value_counts())
print('-'*50)
print('Compare Disciplines for DF_DETAIL (without duplicates)\n')
print(df_detail_no_dups.compare_discipline.value_counts())
df_detail.final_finding.unique()
#counting the number and percentage of 'SU' findings in data set with duplicate data (recommneded finding)
smthCount = 0
for ind in df_detail.index:
    recFin = df_detail['recommended_finding'][ind]
    if recFin == 'SU':
        smthCount += 1
totLen = len(df_detail['recommended_finding'].notnull())
print('Recommended Finding Data (with Duplicates)\n')
print('Number of Findings that were \'Something\': '+ str(smthCount))
print('Number of Total Findings: ' + str(totLen))
print('Percentage of \'Something\' Findings: '+str(smthCount/totLen*100)+'%')
#counting the number and percentage of 'SU' findings in data set with duplicate data (final finding)
smthCount = 0
for ind in df_detail.index:
    recFin = df_detail['final_finding'][ind]
    if recFin == 'SU':
        smthCount += 1
totLen = len(df_detail['final_finding'].notnull())
        
print('Final Finding Data (with Duplicates)\n')
print('Number of Findings that were \'Something\': '+ str(smthCount))
print('Number of Total Findings: ' + str(totLen))
print('Percentage of \'Something\' Findings: '+str(smthCount/totLen*100)+'%')
#counting the number and percentage of 'SU' findings in data set without duplicate data (recommneded finding)
smthCount = 0
for ind in df_detail_no_dups.index:
    recFin = df_detail_no_dups['recommended_finding'][ind]
    if recFin == 'SU':
        smthCount += 1
totLen = len(df_detail_no_dups['recommended_finding'].notnull())

print('Recommended Finding Data (without Duplicates)\n')
print('Number of Findings that were \'Something\': '+ str(smthCount))
print('Number of Total Findings: ' + str(totLen))
print('Percentage of \'Something\' Findings: '+str(smthCount/totLen*100)+'%')
#counting the number and percentage of 'SU' findings in data set without duplicate data (final finding)
smthCount = 0
for ind in df_detail_no_dups.index:
    recFin = df_detail_no_dups['final_finding'][ind]
    if recFin == 'SU':
        smthCount += 1
totLen = len(df_detail_no_dups['final_finding'].notnull())

print('Final Finding Data (without Duplicates)\n')
print('Number of Findings that were \'Something\': '+ str(smthCount))
print('Number of Total Findings: ' + str(totLen))
print('Percentage of \'Something\' Findings: '+str(smthCount/totLen*100)+'%')
#implementing a 'different' coutner for final and recommended finding on data with duplicates
diffCount = 0
for ind in df_detail.index:
    recFin = df_detail['recommended_finding'][ind]
    finFin = df_detail['final_finding'][ind]
    if recFin == 'SU' and finFin != 'SU' and pd.notnull(finFin):
        diffCount += 1
    if finFin == 'SU' and recFin != 'SU' and pd.notnull(recFin):
        diffCount+=1
totLen = len(df_detail['recommended_finding'].notnull())
print('Finding Comparison Data (with Duplicates)')
print('Number of Incidnets where Recommended and Final Findings Do Not Coincide: '+str(diffCount))
print('Total Number of Incidents: '+str(totLen))
print('Percentage of Differenct Recommended and Final Findings: '+str(diffCount/totLen*100)+'%')
#implementing a 'different' coutner for final and recommended finding on data without duplicates
diffCount = 0
for ind in df_detail_no_dups.index:
    recFin = df_detail_no_dups['recommended_finding'][ind]
    finFin = df_detail_no_dups['final_finding'][ind]
    if recFin == 'SU' and finFin != 'SU' and pd.notnull(finFin):
        diffCount += 1
    if finFin == 'SU' and recFin != 'SU' and pd.notnull(recFin):
        diffCount+=1
totLen = len(df_detail_no_dups['recommended_finding'].notnull())
print('Finding Comparison Data (without Duplicates)')
print('Number of Incidnets where Recommended and Final Findings Do Not Coincide: '+str(diffCount))
print('Total Number of Incidents: '+str(totLen))
print('Percentage of Differenct Recommended and Final Findings: '+str(diffCount/totLen*100)+'%')
#helper func to compare incidents when final and recommended finding didn't coincide
noneList = ['UN', 'NS', 'EX', 'NAF', 'NC']
def compareFinding (rec, fin):
    if pd.isnull(rec) or pd.isnull(fin):
        return 'NO DATA'
    if rec == 'SU' and fin != 'SU':
        return 'LESS THAN RECOMMENDED'
    if rec != 'SU' and fin == 'SU':
        return 'GREATER THAN RECOMMENDED'
    if rec == fin or (rec in noneList and fin in noneList):
        return 'EQUAL'
#apply aboce func to both datasets
df_detail['compare_findings'] = df_detail.apply(lambda row: compareFinding(row['recommended_finding'], row['final_finding']), axis = 1)
df_detail_no_dups['compare_findings'] = df_detail_no_dups.apply(lambda row: compareFinding(row['recommended_finding'], row['final_finding']), axis = 1)
#print a counter of finding comparisons for both dataframes
print('Finding Data on Differences between Recommended and Final Findings (with Duplicates)\n')
print(df_detail.compare_findings.value_counts())
print('-'*50)
print('Finding Data on Differences between Recommended and Final Findings (without Duplicates)\n')
print(df_detail_no_dups.compare_findings.value_counts())
print('-'*50)
df_detail.set_index('cr_id', inplace = True)
df_merged = df_main.merge(df_victims, how='outer', left_index=True, right_index=True)
df_merged = df_merged.merge(df_detail, how = 'outer', left_index = True, right_index = True)
df_merged.head()
#remove all rows that have a null value
df_merged_match = df_merged.dropna(axis = 0, how = 'any')
#see how many actual incidents we are working with
len(df_merged_match)
#create counter for most common beats in merged
merged_match_beats = df_merged_match.beat.value_counts()
merged_match_beats
#graph above counter with top 20 beats
sns.set(font_scale = 1)
ax = sns.barplot(x = merged_match_beats[:20].index, y = merged_match_beats[:20], order = merged_match_beats[:20].index, label = 'small')
ax.set_title('20 Beats with HIGHEST Number of Complaint Reports (Merged Data)')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
ax.set(xlabel='Beats', ylabel='Number of Complaints')
ax.tick_params(labelsize=8.5)
plt.show()
#break down top 20 beats into frequency by district
districts_high = {}
for beat in merged_match_beats[:20].index:
    district = beat[:2]
    if district in districts_high.keys():
        districts_high[district] += 1
    else:
        districts_high[district] = 1
#print(districts_high)
districts_high_sorted_keys = sorted(districts_high, key=districts_high.get, reverse=True)
districts_high_sort ={}
for district in districts_high_sorted_keys:
    districts_high_sort[district] = districts_high[district]
print(districts_high_sort)
#graph above created district data
sns.set(font_scale = 1)
ax = sns.barplot(x = list(districts_high_sort.keys()), y = list(districts_high_sort.values()), order = list(districts_high_sort.keys()), label = 'small')
ax.set_title('Districts with Most Beats in \'Top 20 Highest Number of Complaint Reports (Merged Data)\'')
ax.set_xticklabels(ax.get_xticklabels())
ax.set(xlabel='Beats', ylabel='Number of Complaints')
ax.tick_params(labelsize=8.5)
plt.show()
#race counter for merged data
df_merged_match.race.value_counts()
#graph of above counter
ax = sns.countplot(x="race", data=df_merged_match, order = df_merged_match.race.value_counts().index)
ax.set_title('Breakdown of Complainees by Race (Merged Data)')
ax.set_xticklabels(ax.get_xticklabels(),rotation=90)
plt.show()
#gender counter for merged data
df_merged_match.gender.value_counts()
#graph of above counter
ax = sns.countplot(x="gender", data=df_merged_match, order = df_merged_match.gender.value_counts().index)
ax.set_title('Breakdown of Complainees by Gender (Merged Data)')
ax.set_xticklabels(ax.get_xticklabels())
plt.show()
#age range counter for merged data
df_merged_match.age_band.value_counts()
#graph of above counter
age_order = ['<20', '20-29', '30-39', '40-49', '50-59', '60-69','>=70']
ax = sns.countplot(x="age_band", data=df_merged_match, order = age_order)
ax.set_title('Breakdown of Complainees by Age (Merged Data)')
ax.set_xticklabels(ax.get_xticklabels())
plt.show()
#create counter for most common complaint types in merged data
print('20 Most Frequent Complaint Types (Merged Data) — Table\n')
for key in df_merged_match.complaint_key.value_counts()[:20].index:
    if(key == 'OTHER'):
        print('OTHER')
        print("Frequency: "+str(df_merged_match.complaint_key.value_counts()[key]))
        print('-'*50)
    else:
        print(key+": "+complaint_cat_dict[key])
        print("Frequency: "+str(df_merged_match.complaint_key.value_counts()[key]))
        print('-'*50)
#create counter for least common complaint types in merged data
print('20 Least Frequent Complaint Types (Merged Data) — Table\n')
for key in df_merged_match.complaint_key.value_counts()[-20:].index:
    if(key == 'OTHER'):
        print('OTHER')
        print("Frequency: "+str(df_merged_match.complaint_key.value_counts()[key]))
        print('-'*50)
    else:
        print(key+": "+complaint_cat_dict[key])
        print("Frequency: "+str(df_merged_match.complaint_key.value_counts()[key]))
        print('-'*50)
df_merged_match.reset_index(level =0, inplace = True)
#counter for recommended disciplines
discipline_dict_rec = {}
discipline_dict_rec['x Day Suspension'] = 0
discipline_dict_rec['Over 30 Day Suspension'] = 0
discipline_dict_rec['Reprimanded'] = 0
discipline_dict_rec['Administrative Termination'] = 0
discipline_dict_rec['Year-long Suspension or Longer'] = 0
discipline_dict_rec['Seperation'] = 0
discipline_dict_rec['Reinstated'] = 0
discipline_dict_rec['Nothing'] = 0
discipline_dict_rec['Resigned'] = 0
for ind in df_merged_match.index:
    rec = df_merged_match['recommended_discipline'][ind]
    if rec <199:
        discipline_dict_rec['x Day Suspension'] += 1
    if rec == 200:
        discipline_dict_rec['Over 30 Day Suspension'] +=1
    if rec == 0 or rec == 100:
        discipline_dict_rec['Reprimanded'] += 1
    if rec == 300:
        discipline_dict_rec['Administrative Termination'] +=1
    if rec == 365:
        discipline_dict_rec['Year-long Suspension or Longer'] +=1
    if rec == 400:
        discipline_dict_rec['Seperation'] += 1
    if rec == 500:
        discipline_dict_rec['Reinstated'] += 1
    if rec == 600 or rec == 900:
        discipline_dict_rec['Nothing'] += 1
    if rec == 800:
        discipline_dict_rec['Resigned'] += 1
discipline_dict_rec_keys = sorted(discipline_dict_rec, key=discipline_dict_rec.get, reverse=True)
discipline_dict_rec_sort ={}
for district in discipline_dict_rec_keys:
    discipline_dict_rec_sort[district] = discipline_dict_rec[district]
print(discipline_dict_rec_sort)
#counter for final disciplines
discipline_dict_fin = {}
discipline_dict_fin['x Day Suspension'] = 0
discipline_dict_fin['Over 30 Day Suspension'] = 0
discipline_dict_fin['Reprimanded'] = 0
discipline_dict_fin['Administrative Termination'] = 0
discipline_dict_fin['Year-long Suspension or Longer'] = 0
discipline_dict_fin['Seperation'] = 0
discipline_dict_fin['Reinstated'] = 0
discipline_dict_fin['Nothing'] = 0
discipline_dict_fin['Resigned'] = 0
for ind in df_merged_match.index:
    fin = df_merged_match['final_discipline'][ind]
    if fin <199:
        discipline_dict_fin['x Day Suspension'] += 1
    if fin == 200:
        discipline_dict_fin['Over 30 Day Suspension'] +=1
    if fin == 0 or fin == 100:
        discipline_dict_fin['Reprimanded'] += 1
    if fin == 300:
        discipline_dict_fin['Administrative Termination'] +=1
    if fin == 365:
        discipline_dict_fin['Year-long Suspension or Longer'] +=1
    if fin == 400:
        discipline_dict_fin['Seperation'] += 1
    if fin == 500:
        discipline_dict_fin['Reinstated'] += 1
    if fin == 600 or fin == 900:
        discipline_dict_fin['Nothing'] += 1
    if fin == 800:
        discipline_dict_fin['Resigned'] += 1
discipline_dict_fin_keys = sorted(discipline_dict_fin, key=discipline_dict_fin.get, reverse=True)
discipline_dict_fin_sort ={}
for district in discipline_dict_fin_keys:
    discipline_dict_fin_sort[district] = discipline_dict_fin[district]
print(discipline_dict_fin_sort)
#view all discipline data for merged data set + light analysis
print('DISCIPLINE DATA (Merged Data)\n')
for disc in discipline_dict_rec_sort:
    print(disc)
    print('Frequency that '+disc+' was the Recommended Discipline: '+str(discipline_dict_rec[disc]))
    print('Frequency that '+disc+' was the Final Discipline: '+str(discipline_dict_fin[disc]))
    if discipline_dict_rec[disc]>discipline_dict_fin[disc]:
        print('Difference between Recommended and Final Frequencies: '+str(discipline_dict_rec[disc]-discipline_dict_fin[disc])+' ('+ str(abs(discipline_dict_rec[disc]-discipline_dict_fin[disc]))+' more Recommended than Final)')
    elif discipline_dict_rec[disc]<discipline_dict_fin[disc]:
        print('Difference between Recommended and Final Frequencies: '+str(discipline_dict_rec[disc]-discipline_dict_fin[disc])+' ('+ str(abs(discipline_dict_rec[disc]-discipline_dict_fin[disc]))+' more Final than Recommended)')
    else:
        print('Frequencies of Recommended and Final are Equal')
    print('-'*50)
#view of discpline comparison counter
print('Compare Disciplines for DF_DETAIL (with duplicates)\n')
print(df_merged_match.compare_discipline.value_counts())
#counter and view for findings (recommended) in merged
smthCount = 0
for ind in df_merged_match.index:
    recFin = df_merged_match['recommended_finding'][ind]
    if recFin == 'SU':
        smthCount += 1
totLen = len(df_merged_match['recommended_finding'].notnull())
print('Recommended Finding Data (Merged Data)\n')
print('Number of Findings that were \'Something\': '+ str(smthCount))
print('Number of Total Findings: ' + str(totLen))
print('Percentage of \'Something\' Findings: '+str(smthCount/totLen*100)+'%')
#counter and view for findings (final) in merged
smthCount = 0
for ind in df_merged_match.index:
    recFin = df_merged_match['final_finding'][ind]
    if recFin == 'SU':
        smthCount += 1
totLen = len(df_merged_match['final_finding'].notnull())
print('Final Finding Data (Merged Data)\n')
print('Number of Findings that were \'Something\': '+ str(smthCount))
print('Number of Total Findings: ' + str(totLen))
print('Percentage of \'Something\' Findings: '+str(smthCount/totLen*100)+'%')
#counter and view for the differences in recommended and final finding
diffCount = 0
for ind in df_merged_match.index:
    recFin = df_merged_match['recommended_finding'][ind]
    finFin = df_merged_match['final_finding'][ind]
    if recFin == 'SU' and finFin != 'SU' and pd.notnull(finFin):
        diffCount += 1
    if finFin == 'SU' and recFin != 'SU' and pd.notnull(recFin):
        diffCount+=1
totLen = len(df_merged_match['recommended_finding'].notnull())
print('Finding Comparison Data (Merged Data)')
print('Number of Incidnets where Recommended and Final Findings Do Not Coincide: '+str(diffCount))
print('Total Number of Incidents: '+str(totLen))
print('Percentage of Differenct Recommended and Final Findings: '+str(diffCount/totLen*100)+'%')
#counter of comparisons on findings for merged data
print('Finding Data on Differences between Recommended and Final Findings (Merged Data)\n')
print(df_merged_match.compare_findings.value_counts())