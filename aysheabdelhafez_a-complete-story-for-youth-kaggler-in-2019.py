

import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



import pandas as pd

from pandas import Series,DataFrame

import numpy as np

import seaborn as sns

import matplotlib.pyplot as plt

import warnings



warnings.filterwarnings('ignore')

%matplotlib inline
file_path="/kaggle/input/kaggle-survey-2019/multiple_choice_responses.csv"

kaggle_O=pd.read_csv(file_path)

kaggle_O=kaggle_O[1:]

kaggle_O.Q1=kaggle_O.Q1.astype("category")

kaggle_O["Age_generation"]=kaggle_O.Q1

kaggle_O["Age_generation"]=kaggle_O["Age_generation"].replace(["35-39","40-44","45-49"],"Adults")

kaggle_O["Age_generation"]=kaggle_O["Age_generation"].replace(["18-21","22-24","25-29","30-34"],"Millennials")

kaggle_O["Age_generation"]=kaggle_O["Age_generation"].replace(["50-54","55-59","60-69","70+"],"elderly")



kaggle_O.Q3=kaggle_O.Q3.astype("category")

kaggle_O["continent"] =kaggle_O["Q3"]

kaggle_O["continent"]= kaggle_O["continent"].replace(["Malaysia","Philippines","Saudi Arabia",'Iran, Islamic Republic of...',"Viet Nam",'Republic of Korea',"India", 'Russia',"Greece","Pakistan","Japan","South Korea","Indonesia",'Hong Kong (S.A.R.)',"Turkey","Singapore","Israel","Taiwan",'Bangladesh','Thailand', 'China'],"Asia")

kaggle_O["continent"]= kaggle_O["continent"].replace(["Algeria",'Tunisia',"Nigeria","Morocco","South Africa","Egypt","Kenya"],"Africa")

kaggle_O["continent"]= kaggle_O["continent"].replace(['Austria',"Belgium","Romania",'Denmark','France','Germany','Netherlands','Italy','Ireland',"Ukraine",'Portugal',"Switzerland","Chile","Hungary","Norway","Belarus","Sweden","United Kingdom of Great Britain and Northern Ireland","Poland","Spain","Czech Republic"],"Europe")

kaggle_O["continent"]= kaggle_O["continent"].replace(["United States of America","Mexico","Canada"],"North America")

kaggle_O["continent"]= kaggle_O["continent"].replace(["Brazil","Argentina","Colombia",'Peru'],"South America")

kaggle_O["continent"]= kaggle_O["continent"].replace(['Australia',"New Zealand"],"'Australia")

kaggle_O["continent"]= kaggle_O["continent"].replace(["Other"],"Not classified")



kaggle_O.Q5=kaggle_O.Q5.astype("category")

kaggle_O["Employed"]=kaggle_O["Q5"]

kaggle_O["Employed"]=kaggle_O["Employed"].replace(["Data Scientist","Software Engineer","Other","Data Analyst","Research Scientist",

                                               "Business Analyst","Product/Project Manager","Data Engineer","Statistician",

                                               "DBA/Database Engineer"],"Employed")



kaggle_O.Q4=kaggle_O.Q4.astype("category")

kaggle_O["Educated"]=kaggle_O["Q4"]

kaggle_O["Educated"]=kaggle_O["Educated"].replace(["Master’s degree","Bachelor’s degree","Doctoral degree","Professional degree "],"Educated")
kaggle=kaggle_O[(kaggle_O.Age_generation=="Millennials")]

plt.figure(figsize=(10,8))

ax1=sns.countplot(y="Q2",data=kaggle, order=kaggle["Q2"].value_counts().index)

plt.title('Figure(1):Gender Distribution of Youth kaggle participants in 2019')

plt.xlabel("Percentage")

plt.ylabel('Gender')



for p in ax1.patches:

    ax1.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')





plt.figure(figsize=(15,15))

plt.subplot(2,1,1)

ax1=sns.countplot(y="Q3",data=kaggle, order=kaggle["Q3"].value_counts()[:10].index)

plt.title('Figure(2.1):Top 10 countries of Youth kaggle participant in 2019')

plt.xlabel("Percentage")

plt.ylabel('Country')



for p in ax1.patches:

    ax1.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')

    

plt.subplot(2,1,2)

ax2=sns.countplot(y="continent",data=kaggle, order=kaggle["continent"].value_counts()[:10].index)



plt.title('Figure(2.2):Continent Distribution of Youth kaggle participant in 2019')

plt.xlabel("Percentage")

plt.ylabel('Continent')



for p in ax2.patches:

    ax2.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')
plt.figure(figsize=(10,10))

ax1=sns.countplot(y="continent",data=kaggle[(kaggle.Q2 == "Female") | (kaggle.Q2 == 'Male')], order=kaggle["continent"].value_counts()[:10].index,hue="Q2")

plt.title('Figure(3):Continent_Gender Distribution of Youth kaggle participant in 2019')

plt.xlabel("Counts")

plt.ylabel('Continent')



for p in ax1.patches:

    ax1.text(p.get_width()+.3, p.get_y()+.2, p.get_width())

    
kaggle["Q4"]=kaggle["Q4"].replace("Some college/university study without earning a bachelor’s degree","college/university study" )

kaggle["Q4"]=kaggle["Q4"].replace("No formal education past high school","high school" )



plt.figure(figsize=(25,10))

plt.subplot(1,2,1)

ax1=sns.countplot(y="Q4",data=kaggle,order=kaggle["Q4"].value_counts().index)

plt.title('Figure(4.1):Distribution of Youth kagglers higher Education degree')

plt.xlabel("Percentage")

plt.ylabel('Education Degree')



for p in ax1.patches:

    ax1.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')



plt.subplot(1,2,2)

ax2=sns.countplot(y="Q4",data=kaggle[(kaggle.Q2 == "Female") | (kaggle.Q2 == 'Male')],order=kaggle["Q4"].value_counts().index,hue="Q2")

plt.title('Figure(4.2):Gender_Education distribution Youth of kaggle participant in 2019')

plt.xlabel("Count")

plt.ylabel('Education status')

for p in ax2.patches:

    ax2.text(p.get_width()+.3, p.get_y()+.2, p.get_width())

plt.figure(figsize=(12,10))

ax=sns.countplot(y="Q5",data=kaggle,order=kaggle["Q5"].value_counts().index)

plt.title('Figure(5):Distribution of Youth kaggler current title')

plt.xlabel("Percentage")

plt.ylabel('Title')



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')


plt.figure(figsize=(27,12))

plt.subplot(1,3,2)

ax2=sns.countplot(y="Employed",data=kaggle[(kaggle.Q2 == "Female") | (kaggle.Q2 == 'Male')],order=kaggle["Employed"].value_counts().index,hue="Q2")

plt.title('Figure(6.1):Employmnet_gender Distribution of Youth kaggle participants for 2019')

plt.xlabel("Counts")

plt.ylabel('Title')

for p in ax2.patches:

    ax2.text(p.get_width()+.3, p.get_y()+.2, p.get_width())







plt.subplot(1,3,3)

ax3=sns.countplot(y="continent",data=kaggle,order=kaggle["continent"].value_counts().index,hue="Employed")

plt.title('Figure(6.2):Continent_Age Distribution of Youth kaggle participants for 2019')

plt.xlabel("Counts")

plt.ylabel('Title')

for p in ax3.patches:

    ax3.text(p.get_width()+.3, p.get_y()+.15, p.get_width())

plt.figure(figsize=(12,5))

ax=sns.countplot(y="Q6",data=kaggle,order=kaggle["Q6"].value_counts().index)

plt.title('Figure(7):Distribution company size for Millennial particpants')

plt.xlabel("Percentage")

plt.ylabel('company size')



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')
plt.figure(figsize=(12,5))

ax=sns.countplot(y="Q7",data=kaggle,order=kaggle["Q7"].value_counts().index)

plt.title("Figure(8):Data science team size for youth kaggle participants in 2019")

plt.xlabel("Percentage")

plt.ylabel('Individual')



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')
plt.figure(figsize=(15,30))

plt.subplot(2,1,1)

ax1=sns.countplot(y="Q8",data=kaggle,order=kaggle["Q8"].value_counts().index)

plt.title("Figure(9.1):Incorporation of machine learning methods into Youth kaggle participant buisness")

plt.xlabel("Percentage")

plt.ylabel('Machine learning Incoperation')



for p in ax1.patches:

    ax1.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')



plt.subplot(2,1,2)

ax2=sns.countplot(x="Q6",data=kaggle,order=kaggle["Q6"].value_counts().index,hue="Q8")

plt.title("Figure(9.2):Incorporation of machine learning methods into Youth kaggle participant buisness according to company size")

plt.xlabel("Company size")

plt.ylabel('Count')

for i in ax2.patches:

    ax2.text(i.get_x()-.005, i.get_height()+.2, i.get_height())

    
Q9         = kaggle.loc[:,"Q9_Part_1":"Q9_Part_8"].apply(lambda x: x.count()).to_frame()

Q9.index   = (["Analyze and understand data to influence product or business decisions",

               "Build and/or run the data infrastructure that my business uses for storing analyzing, and operationalizing data",

               "Build prototypes to explore applying machine learning to new areas',",

                "Build and/or run a machine learning service that operationally improves my product or workflows",

                "Experimentation and iteration to improve existing ML models",

                "Do research that advances the state of the art of machine learning",

                "None of these activities are an important part of my role at work","Other"])

Q9.columns = ["Frequency"]

Q9.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q9)



Q9.plot.barh(width=0.9,figsize=(12, 13))

plt.gca().invert_yaxis()

plt.title('Figure(10): Top ML Activities that Youth kaggle participant adapt in their work ')

plt.show()

plt.figure(figsize=(18,20))

plt.subplot(1,2,1)

ax=sns.countplot(y="Q10",data=kaggle,order=kaggle["Q10"].value_counts().index)

plt.title("Figure(11.1):Yearly compensation for Youth kaggle participant in 2019")

plt.xlabel("Percentage")

plt.ylabel('Compensation')



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')



plt.subplot(1,2,2)

ax=sns.countplot(y="Q10",data=kaggle[(kaggle.Q2 == "Female") | (kaggle.Q2 == 'Male')],order=kaggle["Q10"].value_counts().index,hue="Q2")

plt.title("Figure(11.2):Yearly compensation for Youth kaggle participant in 2019 According to their gender")

plt.xlabel("Counts")



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38,p.get_width())
kaggle["compensation"]=kaggle["Q10"]

kaggle["compensation"]=kaggle["compensation"].replace(["$0-999"],"0-1k")

kaggle["compensation"]=kaggle["compensation"].replace(["1,000-1,999","2,000-2,999","3,000-3,999","4,000-4,999","5,000-7,499",

                                                      "7,500-9,999","8,000-8,999","9,000-9,999"],"1k-10k")



kaggle["compensation"]=kaggle["compensation"].replace(["10,000-14,999","15,000-19,999","20,000-24,999","25,000-29,999","30,000-39,999",

                                                       "40,000-49,999","50,000-59,999","60,000-69,999","70,000-79,999","80,000-89,999","90,000-99,999"],"1k-10k")



kaggle["compensation"]=kaggle["compensation"].replace(["100,000-124,999","125,000-149,999","150,000-199,999","200,000-249,999"

                                                       ,"250,000-299,999","300,000-500,000","> $500,000"],"+100k")



plt.figure(figsize=(15,15))

ax=sns.countplot(x="continent",data=kaggle,order=kaggle["continent"].value_counts().index,hue="compensation")

plt.title("Figure(12):Continent distribution for yearly compensation for youth kaggle participant in 2019")

plt.xlabel("Counts")

plt.ylabel('Compensation')

for i in ax.patches:

    ax.text(i.get_x()-.005, i.get_height()+.2, i.get_height())

plt.figure(figsize=(12,5))

ax=sns.countplot(y="Q11",data=kaggle,order=kaggle["Q11"].value_counts().index)

plt.title("Figure(13): Youth kaggle participant companies total spending on ML and/or cloud computing products at their work in the past 5 years?")

plt.ylabel('Spending')

plt.xlabel("Percentage")



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')
plt.figure(figsize=(15,10))

ax=sns.countplot(x="Q7",order=kaggle["Q7"].value_counts().index,hue="Q11",data=kaggle)

plt.title("Figure(14.1):Youth kaggle participant total spending on machine learning and/or cloud computing products at your work in the past 5 years According to their DS team size")

plt.xlabel("Percentage")

plt.ylabel('Spending')

for i in ax.patches:

    ax.text(i.get_x()-.03, i.get_height()+.5, i.get_height())





plt.figure(figsize=(15,10))

ax=sns.countplot(x="Q6",order=kaggle["Q6"].value_counts().index,hue="Q11",data=kaggle)

plt.title("Figure(14.2):Youth kaggle participant comapnies total spending on ML and/or cloud computing products at their work in the past 5 years according to company size")

plt.xlabel("DS Team Size")

plt.ylabel('Spending')



for i in ax.patches:

    ax.text(i.get_x()-.03, i.get_height()+.5, i.get_height())





plt.figure(figsize=(15,20))

ax=sns.countplot(y="continent",order=kaggle["continent"].value_counts().index,hue="Q11",data=kaggle)

plt.title("Figure(14.3):Youth kaggle participant companies total spending on ML and/or cloud computing products at work in  each continent")

plt.xlabel("Percentage")

plt.ylabel('Spending')



for p in ax.patches:

    ax.text(p.get_width()+.2, p.get_y()+.1,p.get_width())
Q12         = kaggle.loc[:,"Q12_Part_1":"Q12_Part_12"].apply(lambda x: x.count()).to_frame()

Q12.index   = ["Twitter","Hacker News","Reddit","Kaggle","Course Forums","YouTube","Podcasts","Blogs","Journal Publications","Slack Communities","None","other"]

Q12.columns = ["Frequency"]

Q12.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q12)



Q12.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure(15):Youth kaggle participnats favorite media sources that report for data science topics in 2019 ')

plt.show()
Q13         = kaggle.loc[:,"Q13_Part_1":"Q13_Part_12"].apply(lambda x: x.count()).to_frame()

Q13.index   = ["Udacity","Coursera","edX","DataCamp","DataQuest","Kaggle Courses","Fast.ai","Udemy","LinkedIn Learning",

               "University Courses","None","Other"]

Q13.columns = ["Frequency"]

Q13.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q13)



Q13.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure(16): Youth kaggle participant platforms they begun or completed data science courses?')

plt.show()

plt.figure(figsize=(12,5))

ax=sns.countplot(y="Q14",data=kaggle,order=kaggle["Q14"].value_counts().index)

plt.title("Figure(17):Youth kaggle participant primary tool used at work or school to analyze data?")

plt.xlabel("Percentage")

plt.ylabel('Tools')



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')
plt.figure(figsize=(28,12))

plt.subplot(1,2,1)

ax1=sns.countplot(y="Q15",data=kaggle,order=kaggle["Q15"].value_counts().index)

plt.title("Figure(18.1):Youth kaggle participant Years of Experience in writing codes")

plt.xlabel("Percentage")

plt.ylabel("Years of Expereince")



for p in ax1.patches:

    ax1.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')



plt.subplot(1,2,2)

ax2=sns.countplot(y="Q15",data=kaggle[(kaggle.Q2=="Female")|(kaggle.Q2=="Male")],order=kaggle["Q15"].value_counts().index,hue="Q2")

plt.title("Figure(18.2):Gender_ Experience distribution ")

plt.xlabel("Counts")

plt.ylabel("Years of Expereince")



for p in ax2.patches:

    ax2.text(p.get_width()+.3, p.get_y()+.38, p.get_width())

Q16         = kaggle.loc[:,"Q16_Part_1":"Q16_Part_12"].apply(lambda x: x.count()).to_frame()

Q16.index   = ["Jupyter","Rstudio","PyCharm","Atom","Matlab","Visual Studio","Spyder","Vim / Emacs","Notepad++","Sublime Text","None","Other"]

Q16.columns =["Frequency"]

Q16.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q16)



Q16.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure(19):Integrated development environments (IDE) Youth kaggle participants use on a regular basis?')

plt.show()

Q17         = kaggle.loc[:,"Q17_Part_1":"Q17_Part_12"].apply(lambda x: x.count()).to_frame()

Q17.columns = ["Frequency"]

Q17.index   = ["Kaggle Notebooks","Google Colab","Microsoft Azure Notebooks","Google Cloud Notebook Products",

               "Paperspace/Gradient","FloydHub","Binder/JupyterHub","IBM Watson Studio","Code Ocean","AWS Notebook Products",

               "None","Other"]

Q17.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q17)



Q17.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure(20):Hosted notebook products that Youth kaggle participants use on a regular basis?')

plt.show()

Q18         = kaggle.loc[:,"Q18_Part_1":"Q18_Part_12"].apply(lambda x: x.count()).to_frame()

Q18.columns = ["Frequency"]

Q18.index   = ["Python","R","SQl","C","C++","Java","Javascript","TypeScript","Bash","MATLAB","None","Other"]

Q18.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q18)



Q18.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure(21): Youth kaggle participants programming languages using on a regular basis?')

plt.show()

Q34         = kaggle.loc[:,"Q34_Part_1":"Q34_Part_12"].apply(lambda x: x.count()).to_frame()

Q34.columns = ["Frequency"]

Q34.index   = ["MySQL","PostgresSQL","'SQLite","Microsoft SQL Server","Oracle Database","Microsoft Access",

              "AWS Relational Database Service","AWS DynamoDB","Azure SQL Database","Google Cloud SQL","None","Other"]

Q34.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q34)



Q34.plot.barh( width=0.9, figsize=(12, 8) )

plt.gca().invert_yaxis()

plt.title( 'Figure(22):Relational database products do youth kaggle participants use on a regular basis?' )

plt.show()

plt.figure(figsize=(12,5))

ax=sns.countplot(y="Q19",data=kaggle,order=kaggle["Q19"].value_counts().index)

plt.title("Figure(23):Recommended programming language for Aspiring data analysis")

plt.xlabel("Percentage")

plt.ylabel('Tools')



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')
Q20=kaggle.loc[:,"Q20_Part_1":"Q20_Part_12"].apply(lambda x: x.count()).to_frame()

Q20.columns=["Frequency"]

Q20.index=["Ggplot/ggplot2","Matplotlib","Altair","Shiny","D3.js","Plotly/Plotly Express","Bokeh","Seaborn","Geoplotlib",

           "Leaflet/Folium ","None","Other"]

Q20.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q20)



Q20.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure(24):Data visualization libraries or tools do Youth kaggle participants use on a regular basis?')

plt.show()

Q24         = kaggle.loc[:,"Q24_Part_1":"Q24_Part_12"].apply(lambda x: x.count()).to_frame()

Q24.columns = ["Frequency"]

Q24.index   = ["Linear or Logistic Regression","Decision Trees or Random Forests","Gradient Boosting Machines","Bayesian Approaches",

              "Evolutionary Approaches","Dense Neural Networks","Convolutional Neural Networks","Generative Adversarial Networks",

              "Recurrent Neural Networks","Transformer Networks","None","Other"]

Q24.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q24)



Q24.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure (25): ML algorithms do youth kaggle participants use on a regular basis?')

plt.show()
Q28         =  kaggle.loc[:,"Q28_Part_1":"Q28_Part_12"].apply(lambda x: x.count()).to_frame()

Q28.columns = ["Frequency"]

Q28.index   = ["Scikit-learn","TensorFlow","Keras","RandomForest","Xgboost","PyTorch","Caret","LightGBM",

           "Spark MLib","Fast.ai","None","Other"]

Q28.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q28)



Q28.plot.barh(width=0.9,figsize=(12, 5))

plt.gca().invert_yaxis()

plt.title('Figure(26):ML frameworks do youth kaggle participants use on a regular basis?')

plt.show()

Q21         = kaggle.loc[:,"Q21_Part_1":"Q21_Part_5"].apply(lambda x: x.count()).to_frame()

Q21.columns = ["Frequency"]

Q21.index   = ["CPUs","GPUs","TPUs","None/I do not know","Other"]

Q21.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q21)



Q21.plot.barh(width=0.9,figsize=(12, 4))

plt.gca().invert_yaxis()

plt.title('Figure(27):specialized hardware do youth kaggle participants use on a regular basis?')

plt.show()

plt.figure(figsize=(12,5))

ax=sns.countplot(y="Q22",data=kaggle,order=kaggle["Q22"].value_counts().index)

plt.title('figure(28): Have you ever used a TPU (tensor processing unit)?')

plt.xlabel("Percentage")

plt.ylabel('Gender Distribution')



for p in ax.patches:

    ax.text(p.get_width()+.3, p.get_y()+.38, str(round((p.get_width()/len(kaggle))*100, 2))+'%')
Q25         = kaggle.loc[:,"Q25_Part_1":"Q25_Part_8"].apply(lambda x: x.count()).to_frame()

Q25.columns = ["Frequency"]

Q25.index   = ["Automated data augmentation","Automated feature engineering/selection","Automated model selection",

               "Automated model architecture searches","Automated hyperparameter tuning","Automation of full ML pipelines",

               "None","Other"]

Q25.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q25)



Q25.plot.barh(width=0.9,figsize=(12, 8))

plt.gca().invert_yaxis()

plt.title('Figure(29):Automated Machine Learning')

plt.show()

Q26         = kaggle.loc[:,"Q26_Part_1":"Q26_Part_7"].apply(lambda x: x.count()).to_frame()

Q26.columns = ["Frequency"]

Q26.index   = ["General purpose image/video tools","Image segmentation methods","Object detection methods",

               "Image classification and other general purpose networks","Generative Networks","None","Other"]

Q26.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q26)



Q26.plot.barh(width=0.9,figsize=(12, 5))

plt.gca().invert_yaxis()

plt.title('Figure(30):Computer Vision methods do youth kaggle participants use on a regular basis?')

plt.show()

Q27         = kaggle.loc[:,"Q27_Part_1":"Q27_Part_6"].apply(lambda x: x.count()).to_frame()

Q27.columns = ["Frequency"]

Q27.index   = ["Word embeddings/vectors","Encoder-decorder models","Contextualized embeddings","Transformer language models",

              "None","Other"]

Q27.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q27)



Q27.plot.barh(width=0.9,figsize=(12, 5))

plt.gca().invert_yaxis()

plt.title('Figure(31): Natural Language Processing (NLP) methods do youth kaggle participants use on a regular basis?')

plt.show()



Q29         =  kaggle.loc[:,"Q29_Part_1":"Q29_Part_12"].apply(lambda x: x.count()).to_frame()

Q29.columns =  ["Frequency"]

Q29.index   =  ["Google Cloud Platform", "Amazon Web Services", "Microsoft Azure", "IBM Cloud", "Alibaba Cloud", "Salesforce Cloud",

               "Oracle Cloud", "SAP Cloud", "VMware Cloud", "Red Hat Cloud", "None","Other"]

Q29.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q29)



Q29.plot.barh(width=0.9,figsize=(12, 5))

plt.gca().invert_yaxis()

plt.title('Figure(32):Cloud computing platforms do youth kaggle participants use on a regular basis?')

plt.show()



Q30         = kaggle.loc[:,"Q30_Part_1":"Q30_Part_12"].apply(lambda x: x.count()).to_frame()

Q30.columns = ["Frequency"]

Q30.index   = ["AWS Elastic Compute Cloud (EC2)","Google Compute Engine (GCE)","AWS Lambda","Azure Virtual Machines",

              "Google App Engine","Google Cloud Functions","AWS Elastic Beanstalk ","Google Kubernetes Engine","AWS Batch"

              ,"Azure Container Service","None","Other"]

Q30.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q30)



Q30.plot.barh(width=0.9,figsize=(12, 5))

plt.gca().invert_yaxis()

plt.title('Figure (32):Cloud Computing Products do youth kaggle participants use on a regular basis?')

plt.show()

Q31         = kaggle.loc[:,"Q31_Part_1":"Q31_Part_12"].apply(lambda x: x.count()).to_frame()

Q31.columns = ["Frequency"]

Q31.index   = ["Google BigQuery","AWS Redshift","Databricks","AWS Elastic MapReduce","Teradata",'Microsoft Analysis Services',

              "Google Cloud Dataflow","AWS Athena","AWS Kinesis","Google Cloud Pub/Sub","None","Other"]

Q31.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q31)



Q31.plot.barh(width=0.9,figsize=(12, 5))

plt.gca().invert_yaxis()

plt.title('Figure(33):Big data / analytics products do youth kaggle participants use on a regular basis?')

plt.show()

Q32         = kaggle.loc[:,"Q32_Part_1":"Q32_Part_12"].apply(lambda x: x.count()).to_frame()

Q32.columns = ["Frequency"]

Q32.index   = ["SAS","Cloudera","Azure Machine Learning Studio","Google Cloud Machine Learning Engine","Google Cloud Vision ",

              "Google Cloud Speech-to-Text","Google Cloud Natural Language","RapidMiner","Google Cloud Translation",

              "Amazon SageMaker","None","Other"]

Q32.sort_values(by="Frequency",ascending=False,inplace=True)

print(Q32)



Q32.plot.barh(width=0.9,figsize=(12, 5))

plt.gca().invert_yaxis()

plt.title('Figure(34): Machine Learning products do youth kaggle participants use on a regular basis?')

plt.show()

Q33         = kaggle.loc[ : ,"Q33_Part_1":"Q33_Part_12"].apply( lambda x: x.count()).to_frame()

Q33.columns = ["Frequency"]

Q33.index   = ["Google AutoML","H20 Driverless AI","Databricks AutoML","DataRobot AutoML","Tpot","Auto-Keras","Auto-Sklearn",

               "Auto_ml","Xcessiv","MLbox","None","Other"] 

Q33.sort_values( by = "Frequency", ascending = False, inplace = True)

print(Q33)





Q33.plot.barh( width = 0.9, figsize = (12 , 8) )

plt.gca().invert_yaxis() 

plt.title('Figure(35):Automated machine learning tools (or partial AutoML tools) do youth kaggle participants use on a regular basis?')

plt.show()
