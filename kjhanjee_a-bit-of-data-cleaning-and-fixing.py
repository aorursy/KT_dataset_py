import json
import pandas as pd
import numpy as np
#Read Line Delimited JSON files
contents = open('data.ldjson', "r").read()
contents = contents.encode('utf-8')
contents = contents.decode('ascii','replace').replace('�',"")
contents = contents.replace('}{"uniq_id"','}\n{"uniq_id"')
contents = contents.strip().split("}\n")
keys = ["uniq_id","crawl_timestamp","url","job_title","company_name","city","state","country","post_date","job_description","job_requirements","job_type","job_board","geo","site_name","domain","postdate_yyyymmdd","has_expired","last_expiry_check_date","postdate_in_indexname_format","inferred_city","inferred_state","inferred_country","fitness_score","category","company_description","salary_offered","contact_person","contact_email","contact_phone_number"]
data = dict()
for i in range(len(contents)):
  line = contents[i]
  line = line.replace('}','')
  line = line.replace('{','')
  line = line.strip()
  line = '{'+line+'}'
  content = json.loads(line)
  for key in keys:
	  if key in data.keys():
		  if key in content.keys():
			    data[key].append(content[key])
		  else:
			    data[key].append(None)
	  else:
		    if key in content.keys():
				   data[key] = [content[key]]
		    else:
				   data[key] = [None]

df = pd.DataFrame(data)
#Defining a unique function to get a unique list
def unique(list1):
  unique_list = []
  for x in list1:
    if x not in unique_list:
      unique_list.append(x)
  return unique_list
#Fixing Locations
pd.set_option('mode.chained_assignment', None)
cities = pd.read_csv('list_of_cities_and_towns_in_india-834j.csv')
for index2 in cities.index:
	city = cities.iloc[index2,1]
	state = cities.iloc[index2,2]
	country = "IN"
	if not str(city) == 'nan' and not str(state) == 'nan':
		df['city'].loc[df['city'].str.contains(city,case=False,regex=False,na=False)] = city
		df['state'].loc[df['city'].str.contains(city,case=False,regex=False,na=False)] = state
		df['country'].loc[df['city'].str.contains(city,case=False,regex=False,na=False)] = country
		df['city'].loc[df['country'].str.contains(city,case=False,regex=False,na=False)] = city
		df['state'].loc[df['country'].str.contains(city,case=False,regex=False,na=False)] = state
		df['country'].loc[df['country'].str.contains(city,case=False,regex=False,na=False)] = country
df['country'].loc[df['country']=='India'] = "IN"
#Finding out the right data points for filter
match1 = df['job_description'].str.contains('Customer Care', case = False, regex = False, na = False)
match2 = df['job_description'].str.contains('Voice Process', case = False, regex = False, na = False)
match3 = df['job_description'].str.contains('Customer Service', case = False, regex = False, na = False)
match4 = df['job_description'].str.contains('Tech Support', case = False, regex = False, na = False)
match_final = []
for i in range(len(match1)):
  if match1[i] == True or match2[i] == True or match3[i] == True or match4[i] == True:
    match_final.append(True)
  else:
    match_final.append(False)
#Filtering the Data Frame
match_final = pd.Series(match_final, index = range(len(match_final)))
df2 = df.loc[match_final,:]
df2.index = range(len(df2))
#Fixing Eperience Range and finding out unqiue set of skills
df2['Min_Year_Req'] = None
df2['Max_Year_Req'] = None
skills = []
for index1, row  in df2.iterrows():
 skill = ''
 if not str(row['job_requirements']) == 'nan' and not str(row['job_requirements']) == 'None':
  req_split = (row['job_requirements'].split("|"))
  if len(req_split) > 1:
    exp_split = req_split[0]
    skill_split = req_split[1]
    if len(exp_split) < 30:
      exp_split = exp_split.replace(" years", "")
      exp_split = exp_split.split("-")
      df2.iloc[index1,30] = int(exp_split[0].strip())
      df2.iloc[index1,31] = int(exp_split[1].strip())
    if len(skill_split) < 3000:
      skill_split = skill_split.lower().strip().replace("keywords / skills : ","")
      skill_split = skill_split.split(',')
      if isinstance(skill_split,list):
        if len(skill_split) > 1:
          for k in range(len(skill_split)):
            if len(skill_split[k]) < 50:
              skill = skill_split[k].strip().replace('"','').replace("'",'')
              skills.append(skill)
        else:
            if len(skill_split[0]) < 50:
              skill = skill_split[0].strip().replace('"','').replace("'",'')
              skills.append(skill)
      elif isintance(skills_split,str):
        if len(skill_split) < 50:
          skill = skill_split.strip().replace('"','').replace("'",'')
          skills.append(skill)
  else:
    req_split = req_split[0]
    exp_check = False
    if len(req_split) < 30:
       exp_split = req_split
       if exp_split.find('years') > -1:
        exp_check = True
       else:
        exp_check = False
    else:
      exp_check = False

    if exp_check:  
      exp_split = exp_split.replace(" years", "")
      exp_split = exp_split.split("-")
      df2.iloc[index1,30] = int(exp_split[0].strip())
      df2.iloc[index1,31] = int(exp_split[1].strip())
    else:
       skill_split = req_split[0]
       skill_split = skill_split.lower().strip().replace("keywords / skills : ","")
       skill_split = skill_split.split(',')
       if isinstance(skill_split,list):
        if len(skill_split) > 1:
          for k in range(len(skill_split)):
            if len(skill_split[k]) < 50:
              skill = skill_split[k].strip().replace('"','').replace("'",'')
              skills.append(skill)
        else:
          if len(skill_split[0]) < 50:
            skill = skill_split[0].strip().replace('"','').replace("'",'')
            skills.append(skill)
       elif isinstance(skill_split,str):
         if len(skill_split) < 50:
            skill = skill_split.strip().replace('"','').replace("'",'')
            skills.append(skill)
skills = unique(skills)
for skill in skills:
  df2[skill] = 0
  df2[skill].loc[df2['job_requirements'].str.contains(skill,case=False,regex=False,na=False)] = 1
    
#This will take a lot of time in case the Skills list is not cleaned properly.