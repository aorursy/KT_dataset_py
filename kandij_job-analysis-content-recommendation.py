#importing libraries



import pandas as pd

import numpy as np



# uploading the file from the local drive



#final_jobs = pd.read_csv("C:\\Users\\KANDIRAJU\\Downloads\\neat_Data-20190423T142731Z-001\\neat_Data\\Combined_Jobs_Final.csv")

final_jobs = pd.read_csv("../input/Combined_Jobs_Final.csv")

# listing out the first 5 rows of the data set



final_jobs.head()
# Listing out all the columns that are present in the data set.



list(final_jobs) 
print(final_jobs.shape)

final_jobs.isnull().sum()
#subsetting only needed columns and not considering the columns that are not necessary

cols = list(['Job.ID']+['Slug']+['Title']+['Position']+ ['Company']+['City']+['Employment.Type']+['Education.Required']+['Job.Description'])

final_jobs =final_jobs[cols]

final_jobs.columns = ['Job.ID','Slug', 'Title', 'Position', 'Company','City', 'Empl_type','Edu_req','Job_Description']

final_jobs.head() 
# checking for the null values again.

final_jobs.isnull().sum()
#selecting NaN rows of city

nan_city = final_jobs[pd.isnull(final_jobs['City'])]

print(nan_city.shape)

nan_city.head()
nan_city.groupby(['Company'])['City'].count() 
#replacing nan with thier headquarters location

final_jobs['Company'] = final_jobs['Company'].replace(['Genesis Health Systems'], 'Genesis Health System')



final_jobs.ix[final_jobs.Company == 'CHI Payment Systems', 'City'] = 'Illinois'

final_jobs.ix[final_jobs.Company == 'Academic Year In America', 'City'] = 'Stamford'

final_jobs.ix[final_jobs.Company == 'CBS Healthcare Services and Staffing ', 'City'] = 'Urbandale'

final_jobs.ix[final_jobs.Company == 'Driveline Retail', 'City'] = 'Coppell'

final_jobs.ix[final_jobs.Company == 'Educational Testing Services', 'City'] = 'New Jersey'

final_jobs.ix[final_jobs.Company == 'Genesis Health System', 'City'] = 'Davennport'

final_jobs.ix[final_jobs.Company == 'Home Instead Senior Care', 'City'] = 'Nebraska'

final_jobs.ix[final_jobs.Company == 'St. Francis Hospital', 'City'] = 'New York'

final_jobs.ix[final_jobs.Company == 'Volvo Group', 'City'] = 'Washington'

final_jobs.ix[final_jobs.Company == 'CBS Healthcare Services and Staffing', 'City'] = 'Urbandale'
final_jobs.isnull().sum()
#The employement type NA are from Uber so I assume as part-time and full time

nan_emp_type = final_jobs[pd.isnull(final_jobs['Empl_type'])]

print(nan_emp_type)



#replacing na values with part time/full time

final_jobs['Empl_type']=final_jobs['Empl_type'].fillna('Full-Time/Part-Time')

final_jobs.groupby(['Empl_type'])['Company'].count()

list(final_jobs)
final_jobs["pos_com_city_empType_jobDesc"] = final_jobs["Position"].map(str) + " " + final_jobs["Company"] +" "+ final_jobs["City"]+ " "+final_jobs['Empl_type']+" "+final_jobs['Job_Description']

final_jobs.pos_com_city_empType_jobDesc.head()
#removing unnecessary characters between words separated by space between each word of all columns to make the data efficient

final_jobs['pos_com_city_empType_jobDesc'] = final_jobs['pos_com_city_empType_jobDesc'].str.replace('[^a-zA-Z \n\.]'," ") #removing unnecessary characters

final_jobs.pos_com_city_empType_jobDesc.head()
#converting all the characeters to lower case

final_jobs['pos_com_city_empType_jobDesc'] = final_jobs['pos_com_city_empType_jobDesc'].str.lower() 

final_jobs.pos_com_city_empType_jobDesc.head()
final_all = final_jobs[['Job.ID', 'pos_com_city_empType_jobDesc']]

# renaming the column name as it seemed a bit complicated

final_all = final_jobs[['Job.ID', 'pos_com_city_empType_jobDesc']]

final_all = final_all.fillna(" ")



final_all.head()
print(final_all.head(1))
pos_com_city_empType_jobDesc = final_all['pos_com_city_empType_jobDesc']

#removing stopwords and applying potter stemming

from nltk.corpus import stopwords

from nltk.stem import PorterStemmer

stemmer =  PorterStemmer()

stop = stopwords.words('english')

only_text = pos_com_city_empType_jobDesc.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))

only_text.head()

only_text = only_text.apply(lambda x : filter(None,x.split(" ")))

print(only_text.head())
only_text = only_text.apply(lambda x : [stemmer.stem(y) for y in x])

print(only_text.head())
only_text = only_text.apply(lambda x : " ".join(x))

print(only_text.head())
#adding the featured column back to pandas

final_all['text']= only_text

# As we have added a new column by performing all the operations using lambda function, we are removing the unnecessary column

#final_all = final_all.drop("pos_com_city_empType_jobDesc", 1)



list(final_all)

final_all.head()
# in order to save this file for a backup

#final_all.to_csv("job_data.csv", index=True)


#initializing tfidf vectorizer

from sklearn.feature_extraction.text import TfidfVectorizer

#from sklearn.feature_extraction.text import CountVectorizer



tfidf_vectorizer = TfidfVectorizer()



tfidf_jobid = tfidf_vectorizer.fit_transform((final_all['text'])) #fitting and transforming the vector

tfidf_jobid
#Consider a  new data set and  taking the datasets job view, position of interest, experience of the applicant into consideration for creating a query who applied for job

job_view = pd.read_csv("../input/Job_Views.csv")

job_view.head()

#subsetting only needed columns and not considering the columns that are not necessary as we did that earlier.

job_view = job_view[['Applicant.ID', 'Job.ID', 'Position', 'Company','City']]



job_view["pos_com_city"] = job_view["Position"].map(str) + "  " + job_view["Company"] +"  "+ job_view["City"]



job_view['pos_com_city'] = job_view['pos_com_city'].str.replace('[^a-zA-Z \n\.]',"")



job_view['pos_com_city'] = job_view['pos_com_city'].str.lower()



job_view = job_view[['Applicant.ID','pos_com_city']]



job_view.head()

#Experience

exper_applicant = pd.read_csv("../input/Experience.csv")

exper_applicant.head()
#taking only Position

exper_applicant = exper_applicant[['Applicant.ID','Position.Name']]



#cleaning the text

exper_applicant['Position.Name'] = exper_applicant['Position.Name'].str.replace('[^a-zA-Z \n\.]',"")



exper_applicant.head()

#list(exper_applicant)
exper_applicant['Position.Name'] = exper_applicant['Position.Name'].str.lower()

exper_applicant.head(10)
exper_applicant =  exper_applicant.sort_values(by='Applicant.ID')

exper_applicant = exper_applicant.fillna(" ")

exper_applicant.head(20)

#adding same rows to a single row

exper_applicant = exper_applicant.groupby('Applicant.ID', sort=False)['Position.Name'].apply(' '.join).reset_index()

exper_applicant.head(20)
#Position of interest

poi =  pd.read_csv("../input/Positions_Of_Interest.csv", sep=',')

poi = poi.sort_values(by='Applicant.ID')

poi.head()
# There is no need of application and updation becuase there is no deadline mentioned in the website ( assumption) hence we are droping unimportant attributes

poi = poi.drop('Updated.At', 1)

poi = poi.drop('Created.At', 1)



#cleaning the text

poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.replace('[^a-zA-z \n\.]',"")

poi['Position.Of.Interest']=poi['Position.Of.Interest'].str.lower()

poi = poi.fillna(" ")

poi.head(20)
poi = poi.groupby('Applicant.ID', sort=True)['Position.Of.Interest'].apply(' '.join).reset_index()

poi.head()
#merging jobs and experience dataframes

out_joint_jobs = job_view.merge(exper_applicant, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')

print(out_joint_jobs.shape)

out_joint_jobs = out_joint_jobs.fillna(' ')

out_joint_jobs = out_joint_jobs.sort_values(by='Applicant.ID')

out_joint_jobs.head()
#merging position of interest with existing dataframe

joint_poi_exper_view = out_joint_jobs.merge(poi, left_on='Applicant.ID', right_on='Applicant.ID', how='outer')

joint_poi_exper_view = joint_poi_exper_view.fillna(' ')

joint_poi_exper_view = joint_poi_exper_view.sort_values(by='Applicant.ID')

joint_poi_exper_view.head()
#combining all the columns



joint_poi_exper_view["pos_com_city1"] = joint_poi_exper_view["pos_com_city"].map(str) + joint_poi_exper_view["Position.Name"] +" "+ joint_poi_exper_view["Position.Of.Interest"]



joint_poi_exper_view.head()
final_poi_exper_view = joint_poi_exper_view[['Applicant.ID','pos_com_city1']]

final_poi_exper_view.head()
final_poi_exper_view.columns = ['Applicant_id','pos_com_city1']

final_poi_exper_view.head()
final_poi_exper_view = final_poi_exper_view.sort_values(by='Applicant_id')

final_poi_exper_view.head()
final_poi_exper_view['pos_com_city1'] = final_poi_exper_view['pos_com_city1'].str.replace('[^a-zA-Z \n\.]',"")

final_poi_exper_view.head()

final_poi_exper_view['pos_com_city1'] = final_poi_exper_view['pos_com_city1'].str.lower()

final_poi_exper_view.head()


final_poi_exper_view = final_poi_exper_view.reset_index(drop=True)

final_poi_exper_view.head()
#taking a user

u = 6945

index = np.where(final_poi_exper_view['Applicant_id'] == u)[0][0]

user_q = final_poi_exper_view.iloc[[index]]

user_q
#creating tf-idf of user query and computing cosine similarity of user with job corpus

from sklearn.metrics.pairwise import cosine_similarity

user_tfidf = tfidf_vectorizer.transform(user_q['pos_com_city1'])

output = map(lambda x: cosine_similarity(user_tfidf, x),tfidf_jobid)

output2 = list(output)
#getting the job id's of the recommendations

top = sorted(range(len(output2)), key=lambda i: output2[i], reverse=True)[:50]

recommendation = pd.DataFrame(columns = ['ApplicantID', 'JobID'])

count = 0

for i in top:

    recommendation.set_value(count, 'ApplicantID',u)

    recommendation.set_value(count,'JobID' ,final_all['Job.ID'][i])

    count += 1

recommendation
#getting the job ids and their data

nearestjobs = recommendation['JobID']

job_description = pd.DataFrame(columns = ['JobID','text'])

for i in nearestjobs:

    index = np.where(final_all['Job.ID'] == i)[0][0]    

    job_description.set_value(count, 'JobID',i)

    job_description.set_value(count, 'text', final_all['text'][index])

    count += 1

    
#printing the jobs that matched the query

job_description
job_description.to_csv("recommended_content.csv")
final_all.to_csv("job_data.csv", index=False)