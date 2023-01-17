# # -*- coding: utf-8 -*-
# """
# Original author: Kenarapfaik
# url: https://github.com/arapfaik/scraping-glassdoor-selenium

# Edited by KenJee
# Youtube channel:https://www.youtube.com/channel/UCiT9RITQ9PW6BhXK0y2jaeg
# """
# from selenium.common.exceptions import NoSuchElementException, ElementClickInterceptedException
# from selenium import webdriver
# import time
# import pandas as pd


# def get_jobs(keyword, num_jobs, verbose, path, slp_time):
    
#     '''Gathers jobs as a dataframe, scraped from Glassdoor'''
    
#     #Initializing the webdriver
#     options = webdriver.ChromeOptions()
    
#     #Uncomment the line below if you'd like to scrape without a new Chrome window every time.
#     #options.add_argument('headless')
    
#     #Change the path to where chromedriver is in your home folder.
#     driver = webdriver.Chrome(executable_path=path, options=options)
#     driver.set_window_size(1120, 1000)
    
#     url = "https://www.glassdoor.com/Job/jobs.htm?suggestCount=0&suggestChosen=false&clickSource=searchBtn&typedKeyword="+keyword+"&sc.keyword="+keyword+"&locT=&locId=&jobType="
#     #url = 'https://www.glassdoor.com/Job/jobs.htm?sc.keyword="' + keyword + '"&locT=C&locId=1147401&locKeyword=San%20Francisco,%20CA&jobType=all&fromAge=-1&minSalary=0&includeNoSalaryJobs=true&radius=100&cityId=-1&minRating=0.0&industryId=-1&sgocId=-1&seniorityType=all&companyId=-1&employerSizes=0&applicationType=0&remoteWorkType=0'
#     driver.get(url)
#     jobs = []

#     while len(jobs) < num_jobs:  #If true, should be still looking for new jobs.

#         #Let the page load. Change this number based on your internet speed.
#         #Or, wait until the webpage is loaded, instead of hardcoding it.
#         time.sleep(slp_time)

#         #Test for the "Sign Up" prompt and get rid of it.
#         try:
#             driver.find_element_by_class_name("selected").click()
#         except ElementClickInterceptedException:
#             pass

#         time.sleep(.1)

#         try:
#             driver.find_element_by_css_selector('[alt="Close"]').click() #clicking to the X.
#             print(' x out worked')
#         except NoSuchElementException:
#             print(' x out failed')
#             pass

        
#         #Going through each job in this page
#         job_buttons = driver.find_elements_by_class_name("jl")  #jl for Job Listing. These are the buttons we're going to click.
#         for job_button in job_buttons:  

#             print("Progress: {}".format("" + str(len(jobs)) + "/" + str(num_jobs)))
#             if len(jobs) >= num_jobs:
#                 break

#             job_button.click()  #You might 
#             time.sleep(1)
#             collected_successfully = False
            
#             while not collected_successfully:
#                 try:
#                     company_name = driver.find_element_by_xpath('.//div[@class="employerName"]').text
#                     location = driver.find_element_by_xpath('.//div[@class="location"]').text
#                     job_title = driver.find_element_by_xpath('.//div[contains(@class, "title")]').text
#                     job_description = driver.find_element_by_xpath('.//div[@class="jobDescriptionContent desc"]').text
#                     collected_successfully = True
#                 except:
#                     time.sleep(5)

#             try:
#                 salary_estimate = driver.find_element_by_xpath('.//span[@class="gray salary"]').text
#             except NoSuchElementException:
#                 salary_estimate = -1 #You need to set a "not found value. It's important."
            
#             try:
#                 rating = driver.find_element_by_xpath('.//span[@class="rating"]').text
#             except NoSuchElementException:
#                 rating = -1 #You need to set a "not found value. It's important."

#             #Printing for debugging
#             if verbose:
#                 print("Job Title: {}".format(job_title))
#                 print("Salary Estimate: {}".format(salary_estimate))
#                 print("Job Description: {}".format(job_description[:500]))
#                 print("Rating: {}".format(rating))
#                 print("Company Name: {}".format(company_name))
#                 print("Location: {}".format(location))

#             #Going to the Company tab...
#             #clicking on this:
#             #<div class="tab" data-tab-type="overview"><span>Company</span></div>
#             try:
#                 driver.find_element_by_xpath('.//div[@class="tab" and @data-tab-type="overview"]').click()

#                 try:
#                     #<div class="infoEntity">
#                     #    <label>Headquarters</label>
#                     #    <span class="value">San Francisco, CA</span>
#                     #</div>
#                     headquarters = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Headquarters"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     headquarters = -1

#                 try:
#                     size = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Size"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     size = -1

#                 try:
#                     founded = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Founded"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     founded = -1

#                 try:
#                     type_of_ownership = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Type"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     type_of_ownership = -1

#                 try:
#                     industry = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Industry"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     industry = -1

#                 try:
#                     sector = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Sector"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     sector = -1

#                 try:
#                     revenue = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Revenue"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     revenue = -1

#                 try:
#                     competitors = driver.find_element_by_xpath('.//div[@class="infoEntity"]//label[text()="Competitors"]//following-sibling::*').text
#                 except NoSuchElementException:
#                     competitors = -1

#             except NoSuchElementException:  #Rarely, some job postings do not have the "Company" tab.
#                 headquarters = -1
#                 size = -1
#                 founded = -1
#                 type_of_ownership = -1
#                 industry = -1
#                 sector = -1
#                 revenue = -1
#                 competitors = -1

                
#             if verbose:
#                 print("Headquarters: {}".format(headquarters))
#                 print("Size: {}".format(size))
#                 print("Founded: {}".format(founded))
#                 print("Type of Ownership: {}".format(type_of_ownership))
#                 print("Industry: {}".format(industry))
#                 print("Sector: {}".format(sector))
#                 print("Revenue: {}".format(revenue))
#                 print("Competitors: {}".format(competitors))
#                 print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@")

#             jobs.append({"Job Title" : job_title,
#             "Salary Estimate" : salary_estimate,
#             "Job Description" : job_description,
#             "Rating" : rating,
#             "Company Name" : company_name,
#             "Location" : location,
#             "Headquarters" : headquarters,
#             "Size" : size,
#             "Founded" : founded,
#             "Type of ownership" : type_of_ownership,
#             "Industry" : industry,
#             "Sector" : sector,
#             "Revenue" : revenue,
#             "Competitors" : competitors})
#             #add job to jobs
            
            
#         #Clicking on the "next page" button
#         try:
#             driver.find_element_by_xpath('.//li[@class="next"]//a').click()
#         except NoSuchElementException:
#             print("Scraping terminated before reaching target number of jobs. Needed {}, got {}.".format(num_jobs, len(jobs)))
#             break

#     return pd.DataFrame(jobs)  #This line converts the dictionary object into a pandas DataFrame.
# #Import necessary packages
# import glassdoor_scraper as gs
# import pandas as pd

# path = 'C:/Users/laust/Documents/ds_salary_project/chromedriver'
# df = gs.get_jobs('data scientist',1000,False,path,15)
# df.to_csv('glassdoor_jobs.csv',index=False)
# #Import necessary packages
# import pandas as pd
# df = pd.read_csv('glassdoor_jobs.csv')

# # 1) Average and parse the Salary Estimate column
# #Mark salary estimates with Employer Provided Salary and per hour
# df['hourly']=df['Salary Estimate'].apply(lambda x: 1 if 'per hour' in x.lower() else 0)
# df['employer_provided']=df['Salary Estimate'].apply(lambda x: 1 if 'employer provided salary:' in x.lower() else 0)

# #Remove any rows where Salary Estimate is not present
# df=df[df['Salary Estimate']!= '-1']

# #Remove the (Glassdoor est.) string with a lambda function
# salary = df['Salary Estimate'].apply(lambda x: x.split('(')[0])

# #Replace the dollar sign and K 
# minus_Kd = salary.apply(lambda x: x.replace('K','').replace('$',''))

# #Replace per hour and employer provided salary
# minus_str = minus_Kd.apply(lambda x : x.lower().replace('per hour','').replace('employer provided salary:',''))

# #Create new columns for min,max, and average estimates for salary
# df['min_salary']=minus_str.apply(lambda x: int(x.split('-')[0]))
# df['max_salary']=minus_str.apply(lambda x: int(x.split('-')[1]))

# #Hourly wages to annual salary
# df['min_salary']=df.apply(lambda x: x.min_salary *2 if x.hourly == 1 else x.min_salary,axis=1)
# df['max_salary']=df.apply(lambda x: x.max_salary *2 if x.hourly == 1 else x.max_salary,axis=1)
# df['avg_salary']=(df.min_salary + df.max_salary)/2

# # 2) Parse job description 
# df['python'] = df['Job Description'].apply(lambda x : 1 if 'python' in x.lower() else 0)
# df.python.value_counts()
# df['r'] = df['Job Description'].apply(lambda x : 1 if 'r studio' in x.lower() or 'r-studio' in x.lower() else 0)
# df.r.value_counts()
# df['spark'] = df['Job Description'].apply(lambda x : 1 if 'spark' in x.lower() else 0)
# df.spark.value_counts()
# df['aws'] = df['Job Description'].apply(lambda x : 1 if 'aws' in x.lower() else 0)
# df.aws.value_counts()
# df['excel'] = df['Job Description'].apply(lambda x : 1 if 'excel' in x.lower() else 0)
# df.excel.value_counts()


# # 3) Split the company for the company name only
# # Check for nonexistent ratings
# # Remove the last 3 characters in any ratings >=0 to remove ratings in company name
# df['company_txt']=df.apply(lambda x: x['Company Name'] if x['Rating'] < 0 else x['Company Name'][:-3],axis=1)



# # 4) Extract the state from the Location column
# df['job_state']=df['Location'].apply(lambda x:x.split(',')[1])
# df.job_state.value_counts()



# # 5) Is the location of the job at headquarters?
# df['same_state']=df.apply(lambda x: 1 if x.Location == x.Headquarters else 0, axis=1)



# # 6) Change Founded column to number of years the company has been around
# df['company_age']=df['Founded'].apply(lambda x : x if x<1 else 2020 - x)

# # 7) Simplify titles and seniority

# def title_simplifier(title):
#     if 'data scientist' in title.lower():
#         return 'data scientist'
#     elif 'data engineer' in title.lower():
#         return 'data engineer'
#     elif 'analyst' in title.lower():
#         return 'analyst'
#     elif 'machine learning' in title.lower():
#         return 'mle'
#     elif 'manager' in title.lower():
#         return 'manager'
#     elif 'director' in title.lower():
#         return 'director'
#     else:
#         return 'na'
    
# def seniority(title):
#     if 'sr' in title.lower() or 'senior' in title.lower() or 'lead' in title.lower() or 'principal' in title.lower():
#         return 'senior'
#     elif 'jr' in title.lower() or 'jr.' in title.lower() or 'junior' in title.lower():
#         return 'jr'
#     else:
#         return 'na'
    
# df['job_simplified']=df['Job Title'].apply(title_simplifier)
# df['seniority'] = df['Job Title'].apply(seniority)



# # 8) Fix state LA
# df['job_state']=df.job_state.apply(lambda x: x.strip() if x.strip().lower() != 'los angeles' else 'CA')



# # 9) Job description length
# df['jobdesc_len']= df['Job Description'].apply(lambda x: len(x))



# # 10) Competitor count
# df['num_comp'] = df['Competitors'].apply(lambda x:len(x.split(','))if x!= '-1' else 0)

# # 11) Remove \n s
# df['company_txt']=df.company_txt.apply(lambda x: x.replace('\n',''))


# # 12) Drop unnamed column
# df_out = df.drop(['Unnamed: 0'],axis=1)
# df_out.to_csv('salary_data_cleaned.csv',index=False)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,GridSearchCV
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

df = pd.read_csv('../input/salary-data-cleaned/salary_data_cleaned.csv')
df.head()
df.describe()
df.columns
df.avg_salary.mean()
df.avg_salary.hist()
df.Rating.hist()
df.company_age.hist()
df.boxplot(column=['avg_salary'])
df.boxplot(column='Rating')
df.boxplot(column='jobdesc_len')
df[['company_age','avg_salary','Rating','jobdesc_len']].corr()
cmap = sns.diverging_palette(220,10,as_cmap=True)
sns.heatmap(df[['company_age','avg_salary','Rating','jobdesc_len']].corr(),vmax=.3,center=0, cmap=cmap,square=True,linewidths=.5,cbar_kws={'shrink':.5})
df_cat=df[['Location','Headquarters','Size','Type of ownership','Industry','Sector','Revenue','company_txt','job_state','same_state','python', 'r', 'spark', 'aws', 'excel', 'company_txt', 'job_state',
       'same_state', 'company_age', 'job_simplified', 'seniority']]
for i in df_cat.columns:
    cat_num=df[i].value_counts()
    print('graph for %s: total = %d'% (i,len(cat_num)))
    chart = sns.barplot(x=cat_num.index,y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
    plt.show()
for i in df_cat.columns:
    cat_num=df[i].value_counts()[:20]
    print('graph for %s: total = %d'% (i,len(cat_num)))
    chart = sns.barplot(x=cat_num.index,y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(),rotation=90)
    plt.show()
pd.pivot_table(df,index='job_simplified',values='avg_salary')
pd.pivot_table(df,index=['job_simplified','seniority'],values='avg_salary')
pd.pivot_table(df,index=['job_state'],values='avg_salary').sort_values('avg_salary',ascending=False)
pd.options.display.max_rows
pd.set_option('display.max_rows',None)
pd.pivot_table(df,index=['job_state','job_simplified'],values='avg_salary',aggfunc='count').sort_values('job_state',ascending=False)
pd.pivot_table(df[df.job_simplified == 'data scientist'],index=['job_state'],values='avg_salary').sort_values('avg_salary',ascending=False)
df_pivots=df[['Rating','Industry','Sector','Revenue','num_comp','hourly','employer_provided','python','r','spark','aws','excel','Type of ownership','avg_salary']]
for i in df_pivots.columns:
    print(i)
    print(pd.pivot_table(df_pivots,index=i,values='avg_salary').sort_values('avg_salary',ascending=False))
pd.pivot_table(df_pivots,index='Revenue',columns='python',values='avg_salary',aggfunc='count')
#Choose relevant columns for model
df.columns
df_model = df[['avg_salary','Rating','Size','Type of ownership','Industry','Sector','Revenue','num_comp','hourly','employer_provided','job_state','same_state','company_age','python','spark','aws','excel','job_simplified','seniority','jobdesc_len']]

#Get dummy data
df_dum = pd.get_dummies(df_model)

#Train test split
X = df_dum.drop('avg_salary',axis=1)
y = df_dum.avg_salary.values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

#Multiple Linear Regression. Use statsmodel and sklearn to compare the two linear regressions
X_sm = X = sm.add_constant(X)
model = sm.OLS(y,X_sm)
model.fit().summary()

lm = LinearRegression()
lm.fit(X_train,y_train)

cross_val_score(lm,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)

#Lasso Regression
lm_l = Lasso()
lm_l.fit(X_train,y_train)
cross_val_score(lm_l,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)
alpha = []
error = []
for i in range(1,100):
    alpha.append(i/10)
    lml = Lasso(alpha=i/10)
    error.append(np.mean(cross_val_score(lml,X_train,y_train,scoring='neg_mean_absolute_error',cv=3)
))
err = tuple(zip(alpha,error))
df_err = pd.DataFrame(err,columns=['alpha','error'])
df_err[df_err.error == max(df_err.error)]
plt.plot(alpha,error)

#Random Forest
rf = RandomForestRegressor()
cross_val_score(rf,X_train,y_train,scoring = 'neg_mean_absolute_error',cv=3)

#Tune models using GridSearchCV
parameters = {'n_estimators':range(10,300,10),'criterion':('mse','mae'),'max_features':('auto','sqrt','log2')}
gs = GridSearchCV(rf,parameters,scoring='neg_mean_absolute_error',cv=3)
gs.fit(X_train,y_train)

gs.best_score_
gs.best_estimator_

#Test ensembles
pred_lm = lm.predict(X_test)
pred_lml = lm_l.predict(X_test)
pred_rf = gs.best_estimator_.predict(X_test)

print('MAE for Linear Regression:',mean_absolute_error(y_test,pred_lm))
print('MAE for Lasso Regression:',mean_absolute_error(y_test,pred_lml))
print('MAE for Random Forest Regressor:',mean_absolute_error(y_test,pred_rf))

#See if two models together perform better
print('MAE of Linear Regression combined with random Forest Regressor:',mean_absolute_error(y_test, (pred_lm+pred_rf)/2))
#Not this time