from IPython.display import Image

Image("../input/imgage1/Liver_Img.PNG")
from IPython.display import Image

Image("../input/flow-diagram/Flow_Diagram.PNG")
## Import the required libraries

import numpy as np

import pandas as pd



import matplotlib.pyplot as plt

import seaborn as sns

from matplotlib.colors import ListedColormap



%matplotlib inline



import category_encoders as ce



from sklearn import preprocessing as prep

from sklearn.utils import resample



from sklearn.model_selection import train_test_split as tts

from sklearn.ensemble import RandomForestClassifier

from sklearn import metrics

from sklearn.metrics import confusion_matrix, roc_auc_score



from yellowbrick import ROCAUC



## Ignoring the warnings by messages

import warnings

warnings.filterwarnings("ignore", message="divide by zero encountered in divide")

warnings.filterwarnings("ignore", message="invalid value encountered in true_divide")

warnings.filterwarnings("ignore", message="invalid value encountered in double_scalars")

warnings.filterwarnings("ignore", message='FutureWarning')
patients_df = pd.read_csv("../input/indian-liver-patient-records/indian_liver_patient.csv")

patients_df.head(10)
patients_df.shape
patients_df.info()
label_font_dict = {'family':'calibri','size':18,'color':'coral','style':'italic'}

title_font_dict = {'family':'calibri','size':20,'color':'Blue','style':'italic'}
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,12))

    sns.heatmap(data=pd.DataFrame(patients_df.isnull()),cmap=ListedColormap(sns.color_palette('GnBu',10)))

    plt.xlabel('Columns',fontdict=label_font_dict)

    plt.ylabel('Record Indexes',fontdict=label_font_dict)

    plt.title('Missing Values in the Dataset',fontdict=title_font_dict)

    plt.xticks(color='black',size=12,style='oblique')

    plt.yticks(color='black',size=10,style='oblique')

plt.show()
patients_as_per_age_gender = pd.DataFrame(patients_df.groupby(by=['Age','Gender']).count()['Total_Bilirubin']).reset_index()

patients_as_per_age_gender.columns = ['Age','Gender','Record_Count']

patients_as_per_age_gender.head()
with plt.style.context('seaborn'):

    plt.figure(figsize=(23,10))

    sns.barplot(x='Age',y='Record_Count',hue='Gender',data=patients_as_per_age_gender,palette=sns.color_palette('gist_rainbow_r',2))

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Record Count',fontdict=label_font_dict)

    plt.title('Age & Gender wise Record_Count',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=15,color='black',size=12,style='oblique')

    plt.legend(loc='upper right')
patients_df['Dataset'].unique()
pos_neg_count = patients_df['Dataset'].value_counts().reset_index()

pos_neg_count.columns = ['Class','Patients_Count']

pos_neg_count
patients_df['Label'] = patients_df['Dataset'].apply(lambda val: val if val == 1 else 0)

patients_df.drop(['Dataset'],axis=1,inplace=True)

pos_neg_count = patients_df['Label'].value_counts().reset_index()

pos_neg_count.columns = ['Class','Patients_Count']

pos_neg_count
patients_df['Label'].unique()
with plt.style.context('seaborn'):

    plt.figure(figsize=(5,7))

    sns.barplot(x='Class',y='Patients_Count',data=pos_neg_count,palette='inferno')

    plt.xlabel('Class',fontdict=label_font_dict)

    plt.ylabel('Patients Count',fontdict=label_font_dict)

    plt.title('Positive & Negative Patients Count',fontdict=title_font_dict)

    plt.xticks(ticks=[0,1],labels=['-ve Diagnosed','+ve Diagnosed'],color='black',size=12,style='oblique')

    plt.yticks(rotation=15,color='black',size=10,style='oblique')

plt.show()
pos_neg_patients_as_per_gender = patients_df.groupby(['Gender','Label']).count()['Age'].reset_index()

pos_neg_patients_as_per_gender.columns = ['Gender','Label','Patients_Count']

pos_neg_patients_as_per_gender
with plt.style.context('seaborn'):

    plt.figure(figsize=(5,7))

    sns.barplot(x='Gender',y='Patients_Count',hue='Label',data=pos_neg_patients_as_per_gender,palette=sns.color_palette('PuBu_r',3))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Patients Count',fontdict=label_font_dict)

    plt.title('Female & Male: +ve and -ve Patients Count',fontdict=title_font_dict)

    plt.xticks(ticks=[0,1],labels=['Female','Male'],color='black',size=12,style='oblique')

    plt.yticks(rotation=15,color='black',size=10,style='oblique')

    plt.legend()

plt.show()
patients_df.head()
pos_neg_patients_as_per_age_gender = patients_df.groupby(['Age','Gender','Label']).count()['Total_Bilirubin'].reset_index()

pos_neg_patients_as_per_age_gender.columns = ['Age','Gender','Label','Patients_Count']

pos_neg_patients_as_per_age_gender.head()
with plt.style.context('seaborn'):

    sns.relplot(x='Age',y='Patients_Count',hue='Gender',col='Label',data=pos_neg_patients_as_per_age_gender,size='Patients_Count',sizes=(95,900),

                palette=sns.color_palette('gist_rainbow_r',2),height=5,aspect=3)

plt.show()
with plt.style.context('ggplot'):

    plt.figure(figsize=(15,20))

    sns.scatterplot(x='Age',y='Total_Bilirubin',hue='Label',data=patients_df,hue_order=[1,0])

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Total Bilirubin',fontdict=label_font_dict)

    plt.title('Relationship b/w Total Bilirubin and Liver Disease',fontdict=title_font_dict)

    plt.xticks(ticks=[0,10,20,30,40,50,60,70,80,90],color='black',size=12,style='oblique')

    plt.yticks(rotation=15,color='black',size=10,style='oblique')

plt.show()
with plt.style.context('ggplot'):

    plt.figure(figsize=(15,20))

    fig = sns.scatterplot(x='Age',y='Total_Bilirubin',hue='Label',data=patients_df,hue_order=[1,0])

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Total Bilirubin',fontdict=label_font_dict)

    plt.title('Relationship b/w Total Bilirubin and Liver Disease',fontdict=title_font_dict)

    plt.xticks(ticks=[0,10,20,30,40,50,60,70,80,90],color='black',size=12,style='oblique')

    plt.yticks(rotation=15,color='black',size=10,style='oblique')

    plt.ylim(bottom=0,top=4)

plt.show()
pos_neg_patients_as_per_age = patients_df.groupby(['Age','Label']).count()['Gender'].reset_index()

pos_neg_patients_as_per_age.columns = ['Age','Label','Patients_Count']

pos_neg_patients_as_per_age.head()
with plt.style.context('seaborn'):

    sns.catplot(x='Age',y='Patients_Count',hue='Label',data=pos_neg_patients_as_per_age,kind='point',

                height=5,aspect=3,hue_order=[1,0],palette=sns.color_palette('gist_rainbow_r',2))

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Patients Count',fontdict=label_font_dict)

    plt.title('Relationship b/w Age and Liver Ailment',fontdict=title_font_dict)

    plt.xticks(color='black',size=9.5,style='oblique')

    plt.yticks(rotation=10,color='black',size=10,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,8))

    sns.scatterplot(x='Total_Bilirubin',y='Direct_Bilirubin',hue='Label',

                    hue_order=[0,1],data=patients_df,palette=sns.color_palette('inferno',2))

    plt.xlabel('Total Bilirubin',fontdict=label_font_dict)

    plt.ylabel('Direct Bilirubin',fontdict=label_font_dict)

    plt.title('Does Total Bilirubin & Direct Bilirubin together leads to Liver Ailment?',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
## Close-eye with shorter axes

with plt.style.context('seaborn'):

    plt.figure(figsize=(14,10))

    sns.scatterplot(x='Total_Bilirubin',y='Direct_Bilirubin',hue='Label',

                    hue_order=[0,1],data=patients_df,palette=sns.color_palette('inferno',2))

    plt.ylim(0,3.5)

    plt.xlim(0,6.5)

    plt.xlabel('Total Bilirubin',fontdict=label_font_dict)

    plt.ylabel('Direct Bilirubin',fontdict=label_font_dict)

    plt.title('Does Total Bilirubin & Direct Bilirubin together leads to Liver Ailment?',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
pearson_coeff_r = pd.DataFrame(np.corrcoef(patients_df['Total_Bilirubin'],patients_df['Direct_Bilirubin']))

pearson_coeff_r
pearson_coeff_r_sqr = pearson_coeff_r.applymap(lambda val: np.square(val))

pearson_coeff_r_sqr
patients_df['Unconjugated_bilirubin'] = patients_df['Total_Bilirubin'] - patients_df['Direct_Bilirubin']

patients_df.head()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,8))

    sns.scatterplot(x='Total_Bilirubin',y='Unconjugated_bilirubin',hue='Label',data=patients_df,palette=sns.color_palette('RdPu_r',2))

    plt.xlabel('Total Bilirubin',fontdict=label_font_dict)

    plt.ylabel('Indirect Bilirubin',fontdict=label_font_dict)

    plt.title('Does Total Bilirubin & In-direct Bilirubin together leads to Liver Ailment?',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,9))

    sns.scatterplot(x='Total_Bilirubin',y='Unconjugated_bilirubin',hue='Label',data=patients_df,palette=sns.color_palette('RdPu_r',2))

    plt.xlim(0,8)

    plt.ylim(0,4)

    plt.xlabel('Total Bilirubin',fontdict=label_font_dict)

    plt.ylabel('Indirect Bilirubin',fontdict=label_font_dict)

    plt.title('Does Total Bilirubin & In-direct Bilirubin together leads to Liver Ailment?',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
np.corrcoef(patients_df['Total_Bilirubin'],patients_df['Unconjugated_bilirubin'])
pd.DataFrame(np.corrcoef(patients_df['Total_Bilirubin'],patients_df['Unconjugated_bilirubin'])).applymap(lambda val: np.square(val))
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Total_Bilirubin',data=patients_df,palette=sns.color_palette('cubehelix',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Total Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of Total Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Total_Bilirubin',data=patients_df,palette=sns.color_palette('cubehelix',2))

    plt.ylim(0,30)

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Total Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of Total Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Total_Bilirubin',hue='Label',data=patients_df,palette=sns.color_palette('cubehelix',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Total Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of Total Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Direct_Bilirubin',data=patients_df,palette=sns.color_palette('twilight',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Direct Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of Direct Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Direct_Bilirubin',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Direct Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of Direct Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Unconjugated_bilirubin',data=patients_df,palette=sns.color_palette('twilight',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('In-Direct Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of In-Direct Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Unconjugated_bilirubin',data=patients_df,palette=sns.color_palette('twilight',2))

    plt.ylim(-10,10)

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('In-Direct Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of In-Direct Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,7))

    sns.swarmplot(x='Gender',y='Unconjugated_bilirubin',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('In-Direct Bilirubin',fontdict=label_font_dict)

    plt.title('Distribution of In-Direct Bilirubin in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,8))    

    sns.scatterplot(x='Direct_Bilirubin',y='Unconjugated_bilirubin',hue='Label',data=patients_df,

                    hue_order=[1,0],palette=sns.color_palette('gnuplot2_r',2))

    plt.xlabel('Direct Bilirubin',fontdict=label_font_dict)

    plt.ylabel('Indirect Bilirubin',fontdict=label_font_dict)

    plt.title('Does Direct Bilirubin & In-direct Bilirubin together leads to Liver Ailment?',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,12))    

    sns.scatterplot(x='Direct_Bilirubin',y='Unconjugated_bilirubin',hue='Label',data=patients_df,

                    hue_order=[1,0],palette=sns.color_palette('gnuplot2_r',2))

    plt.xlim(0,5)

    plt.ylim(0,4)

    plt.xlabel('Direct Bilirubin',fontdict=label_font_dict)

    plt.ylabel('Indirect Bilirubin',fontdict=label_font_dict)

    plt.title('Does Direct Bilirubin & In-direct Bilirubin together leads to Liver Ailment?',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

    plt.legend(loc='upper left')

plt.show()
np.corrcoef(patients_df['Direct_Bilirubin'],patients_df['Unconjugated_bilirubin'])
pd.DataFrame(np.corrcoef(patients_df['Direct_Bilirubin'],patients_df['Unconjugated_bilirubin'])).applymap(lambda val: np.square(val))
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,8))

    sns.pointplot(x='Age',y='Unconjugated_bilirubin',hue='Label',data=patients_df,palette=sns.color_palette('gnuplot2_r',2),hue_order=[1,0],ci=False)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Unconjugated Bilirubin',fontdict=label_font_dict)

    plt.title('Trend of Unconjugated Bilirubin across all ages',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(20,10))

    sns.lineplot(x='Age',y='Unconjugated_bilirubin',hue='Gender',data=patients_df,palette=sns.color_palette('gnuplot2_r',2),ci=95)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Unconjugated Bilirubin',fontdict=label_font_dict)

    plt.title('Trend of Unconjugated Bilirubin across all ages',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.legend(loc='upper right')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,6))

    sns.violinplot(x='Age',y='Unconjugated_bilirubin',hue='Label',hue_order=[1,0],

                data=patients_df[patients_df['Gender'] == 'Female'],palette=sns.color_palette('gist_rainbow_r',2))

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,6))

    sns.violinplot(x='Age',y='Direct_Bilirubin',hue='Label',hue_order=[1,0],

                data=patients_df[patients_df['Gender'] == 'Female'],palette=sns.color_palette('gist_rainbow_r',2),ci=False)

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,6))

    sns.violinplot(x='Age',y='Total_Bilirubin',hue='Label',hue_order=[1,0],

                data=patients_df[patients_df['Gender'] == 'Female'],palette=sns.color_palette('gist_rainbow_r',2),ci=False)

plt.show()
female_pos_neg = patients_df[patients_df['Gender'] == 'Female'].groupby(['Age','Label']).count()['Gender'].reset_index()

female_pos_neg.columns = ['Age','Label','Patients_Count']



with plt.style.context('seaborn'):

    plt.figure(figsize=(16,6))

    sns.lineplot(x='Age',y='Patients_Count',hue='Label',data=female_pos_neg,palette=sns.color_palette('gnuplot_r',2),hue_order=[1,0])

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Female Patients',fontdict=label_font_dict)

    plt.title('Trend of Female Patients across all ages',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,8))

    sns.pointplot(x='Age',y='Alkaline_Phosphotase',hue='Gender',data=patients_df,palette=sns.color_palette('gnuplot_r',2),ci=False)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Alkaline Phosphotase',fontdict=label_font_dict)

    plt.title('Trend of Alkaline Phosphotase across all ages in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Alkaline_Phosphotase',data=patients_df,palette=sns.color_palette('twilight',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Alkaline Phosphotase',fontdict=label_font_dict)

    plt.title('Alkaline Phosphotase distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,10))

    sns.swarmplot(x='Gender',y='Alkaline_Phosphotase',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2),hue_order=[0,1])

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Alkaline Phosphotase',fontdict=label_font_dict)

    plt.title('Alkaline Phosphotase distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,10))

    sns.swarmplot(x='Gender',y='Alkaline_Phosphotase',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2),hue_order=[0,1])

    plt.ylim(0,600)

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Alkaline Phosphotase',fontdict=label_font_dict)

    plt.title('Alkaline Phosphotase distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,8))

    sns.pointplot(x='Age',y='Alamine_Aminotransferase',hue='Gender',data=patients_df,palette=sns.color_palette('gnuplot_r',2),ci=False)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Alanine Aminotransferase (ALT)',fontdict=label_font_dict)

    plt.title('Trend of ALT across all ages in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Alamine_Aminotransferase',data=patients_df,palette=sns.color_palette('twilight',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Alanine Aminotransferase (ALT)',fontdict=label_font_dict)

    plt.title('ALT distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Alamine_Aminotransferase',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Alanine Phosphotase',fontdict=label_font_dict)

    plt.title('ALT distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Alamine_Aminotransferase',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.ylim(0,200)

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Alanine Phosphotase (ALT)',fontdict=label_font_dict)

    plt.title('ALT distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,8))

    sns.pointplot(x='Age',y='Aspartate_Aminotransferase',hue='Gender',data=patients_df,palette=sns.color_palette('gnuplot_r',2),ci=False)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Aspartate Aminotransferase (AST)',fontdict=label_font_dict)

    plt.title('Trend of AST across all ages in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Aspartate_Aminotransferase',data=patients_df,palette=sns.color_palette('twilight',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Aspartate Aminotransferase (AST)',fontdict=label_font_dict)

    plt.title('AST distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Aspartate_Aminotransferase',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Aspartate Aminotransferase (AST)',fontdict=label_font_dict)

    plt.title('AST distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Aspartate_Aminotransferase',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.ylim(0,500)

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Aspartate Aminotransferase (AST)',fontdict=label_font_dict)

    plt.title('AST distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(19,8))

    sns.swarmplot(x='Gender',y='Aspartate_Aminotransferase',hue='Label',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.ylim(0,150)

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Aspartate Aminotransferase (AST)',fontdict=label_font_dict)

    plt.title('AST distribution in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,10))

    sns.scatterplot(x='Alamine_Aminotransferase',y='Aspartate_Aminotransferase',hue='Label',data=patients_df,

                    palette=sns.color_palette('gist_rainbow',2),style='Label')

    plt.xlabel('ALT',fontdict=label_font_dict)

    plt.ylabel('AST',fontdict=label_font_dict)

    plt.title('Relationship b/w ALT & AST',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,10))

    sns.scatterplot(x='Alamine_Aminotransferase',y='Aspartate_Aminotransferase',hue='Label',data=patients_df,

                   palette=sns.color_palette('twilight',2),style='Label')

    plt.xlim(0,250)

    plt.ylim(0,400)

    plt.xlabel('ALT',fontdict=label_font_dict)

    plt.ylabel('AST',fontdict=label_font_dict)

    plt.title('Relationship b/w ALT & AST',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

    plt.legend(loc='upper left')
np.corrcoef(x=patients_df['Alamine_Aminotransferase'],y=patients_df['Aspartate_Aminotransferase'])
pd.DataFrame(np.corrcoef(x=patients_df['Alamine_Aminotransferase'],y=patients_df['Aspartate_Aminotransferase'])).applymap(lambda val:np.square(val))
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,10))

    sns.boxenplot(x='Gender',y='Alkaline_Phosphotase',hue='Label',data=patients_df,palette=sns.color_palette('PuBu',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('ALP',fontdict=label_font_dict)

    plt.title('Quantiles of ALP in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,10))

    sns.boxenplot(x='Gender',y='Alamine_Aminotransferase',hue='Label',data=patients_df,palette=sns.color_palette('plasma',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('ALT',fontdict=label_font_dict)

    plt.title('Quantiles of ALT in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,10))

    sns.boxenplot(x='Gender',y='Aspartate_Aminotransferase',hue='Label',data=patients_df,palette=sns.color_palette('twilight',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('AST',fontdict=label_font_dict)

    plt.title('Quantiles of AST in males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
patients_df['AST_ALT_Ratio'] = np.divide(patients_df['Aspartate_Aminotransferase'],patients_df['Alamine_Aminotransferase'])



with plt.style.context('seaborn'):

    plt.figure(figsize=(15,10))

    sns.violinplot(x='Gender',y='AST_ALT_Ratio',hue='Label',data=patients_df,palette=sns.color_palette('plasma',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('AST/ALT',fontdict=label_font_dict)

    plt.title('Distribution of AST/ALT for males and females',fontdict=title_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
sns.pairplot(data=patients_df[['Total_Bilirubin','Direct_Bilirubin','Unconjugated_bilirubin',

                                           'Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase',

                                           'AST_ALT_Ratio','Label']],hue='Label',palette='husl',hue_order=[1,0])

plt.xticks(color='black',size=11,style='oblique')

plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
plt.figure(figsize=(14,8))

fig = sns.heatmap(patients_df[['Total_Bilirubin','Direct_Bilirubin','Unconjugated_bilirubin',

                                           'Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase',

                                           'AST_ALT_Ratio','Label']].corr(),annot=True,cmap='viridis',linecolor='k',linewidths=1)

bottom, top = fig.get_ylim()

fig.set_ylim(bottom + 0.5, top - 0.5)

plt.title('Correlation Matrix using Heatmap',fontdict=title_font_dict)

plt.xticks(color='black',size=11,style='oblique')

plt.yticks(rotation=0,color='black',size=11,style='oblique')

plt.show()
min_var_vals = patients_df.groupby(['Gender', 'Label']).agg(

    min_Tot_Bilirubin = ('Total_Bilirubin', min),

    min_Dir_Bilirubin = ('Direct_Bilirubin', min),

    min_InDir_Bilirubin = ('Unconjugated_bilirubin', min),

    min_ALP = ('Alkaline_Phosphotase', min),

    min_ALT = ('Alamine_Aminotransferase', min),

    min_AST = ('Aspartate_Aminotransferase', min),

    min_AST_ALT_Ratio = ('AST_ALT_Ratio',min))



min_var_vals
with plt.style.context('seaborn'):

    min_var_vals.plot(kind='bar',colormap='twilight',figsize=(16,9))

    plt.xlabel('Gender,Label',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Minimum values of Liver Enzymes and different Bilirubin',fontdict=title_font_dict)

plt.legend()

plt.show()
max_var_vals = patients_df.groupby(['Gender', 'Label']).agg(

    max_Tot_Bilirubin = ('Total_Bilirubin', max),

    max_Dir_Bilirubin = ('Direct_Bilirubin', max),

    max_InDir_Bilirubin = ('Unconjugated_bilirubin', max),

    max_ALP = ('Alkaline_Phosphotase', max),

    max_ALT = ('Alamine_Aminotransferase', max),

    max_AST = ('Aspartate_Aminotransferase', max),

    max_AST_ALT_Ratio = ('AST_ALT_Ratio',max))



max_var_vals
with plt.style.context('seaborn'):

    max_var_vals.plot(kind='bar',colormap='twilight',figsize=(16,9))

    plt.xlabel('Gender,Label',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Maximum values of Liver Enzymes and different Bilirubin',fontdict=title_font_dict)

plt.legend()

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,10))

    sns.violinplot(x='Gender',hue='Label',y='Total_Protiens',data=patients_df,palette=sns.color_palette('gist_rainbow',2))

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Total Protiens',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Distribution of Total Protiens among males and females for both type of cases',fontdict=title_font_dict)

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,10))

    sns.lineplot(x='Age',y='Total_Protiens',hue='Gender',data=patients_df,palette=sns.color_palette('plasma',2),ci=95)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Total Protiens',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Does Age plays a crucial role in decreasing the level of Total Protiens?',fontdict=title_font_dict)

plt.legend()

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,10))

    sns.violinplot(x='Gender',hue='Label',y='Albumin',data=patients_df,palette=sns.color_palette('magma',2),hue_order=[1,0])

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('Albumin',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Distribution of Albumin among males & females for both type of cases',fontdict=title_font_dict)

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,10))

    sns.lineplot(x='Age',y='Albumin',hue='Gender',data=patients_df,palette=sns.color_palette('plasma',2),ci=95)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('Albumin',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Does Age plays a crucial role in decreasing the level of Albumin?',fontdict=title_font_dict)

plt.legend()

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(15,10))

    sns.violinplot(x='Gender',hue='Label',y='Albumin_and_Globulin_Ratio',data=patients_df,palette=sns.color_palette('nipy_spectral',2),hue_order=[1,0])

    plt.xlabel('Gender',fontdict=label_font_dict)

    plt.ylabel('A/G Ratio',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Distribution of A/G Ratio among males & females for both type of cases',fontdict=title_font_dict)

plt.show()
with plt.style.context('seaborn'):

    plt.figure(figsize=(18,10))

    sns.lineplot(x='Age',y='Albumin_and_Globulin_Ratio',hue='Gender',data=patients_df,palette=sns.color_palette('plasma',2),ci=95)

    plt.xlabel('Age',fontdict=label_font_dict)

    plt.ylabel('A/G Ratio',fontdict=label_font_dict)

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=0,color='black',size=11,style='oblique')

    plt.title('Does Age plays a crucial role in decreasing the level of A/G Ratio?',fontdict=title_font_dict)

plt.legend()

plt.show()
with plt.style.context('seaborn'):

    sns.pairplot(data=patients_df[['Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','AST_ALT_Ratio','Label']],hue='Label',palette='husl',hue_order=[1,0])

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
plt.figure(figsize=(8,6))

fig = sns.heatmap(patients_df[['Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','AST_ALT_Ratio','Label']].corr(),

                  annot=True,cmap='coolwarm',linecolor='k',linewidths=0.9)

bottom, top = fig.get_ylim()

fig.set_ylim(bottom + 0.5, top - 0.5)

plt.title('Correlation Matrix using Heatmap',fontdict=title_font_dict)

plt.xticks(rotation=90,color='black',size=11,style='oblique')

plt.yticks(rotation=0,color='black',size=11,style='oblique')

plt.show()
with plt.style.context('seaborn'):

    sns.pairplot(data=patients_df,hue='Label',palette='husl',hue_order=[1,0])

    plt.xticks(color='black',size=11,style='oblique')

    plt.yticks(rotation=10,color='black',size=11,style='oblique')

plt.show()
plt.figure(figsize=(15,8))

fig = sns.heatmap(patients_df.corr(),

                  annot=True,cmap='coolwarm',linecolor='k',linewidths=0.9)

bottom, top = fig.get_ylim()

fig.set_ylim(bottom + 0.5, top - 0.5)

plt.title('Correlation Matrix using Heatmap',fontdict=title_font_dict)

plt.xticks(rotation=90,color='black',size=11,style='oblique')

plt.yticks(rotation=0,color='black',size=11,style='oblique')

plt.show()
patients_df[patients_df['Albumin_and_Globulin_Ratio'].isna()]
patients_df['Albumin_and_Globulin_Ratio'] = pd.DataFrame(patients_df.apply(lambda val: val['Albumin_and_Globulin_Ratio'] if str(val['Albumin_and_Globulin_Ratio']).upper() != 'NAN' 

                                                                           else val['Albumin']/(val['Total_Protiens'] - val['Albumin']),axis=1))



patients_df.head()
from imblearn.over_sampling import RandomOverSampler

from sklearn.utils import resample
patients_df.head()
patients_df['Label'].value_counts()
patients_df_upsample = pd.concat([patients_df[patients_df['Label']==1],resample(patients_df[patients_df['Label'] == 0], n_samples=230, replace=True, random_state=44)],axis=0)

patients_df_upsample.reset_index(drop=True,inplace=True)

patients_df_upsample.head()
def inspect_feature(df_obj,feature_name):

    """

    This function is created for plotting the histograms and box-plots.



    Parameters

    ----------

    df_obj : DataFrame

        Containing feature that needs to be inspected.

    feature_name : str

        Feature that you want to inspect

    scaler : str, optional

        DESCRIPTION. The default is 'ss'.



    Returns

    -------

    None.

    """

    with plt.style.context('seaborn'):

        df_obj[feature_name].plot(kind='hist')

        plt.title('Raw Data')

        plt.show()

        df_obj[feature_name].plot(kind='box')

        plt.title('Raw Data')

        plt.show()

        np.log1p(df_obj[feature_name]).plot(kind='hist')

        plt.title('Log1p Data')

        plt.show()

        np.log1p(df_obj[feature_name]).plot(kind='box')

        plt.title('Log1p Data')

        plt.show()

        pd.DataFrame(fix_outliers(df_obj,feature_name)).plot(kind='hist')

        plt.title('Outliers fixed')

        plt.show()

        pd.DataFrame(fix_outliers(df_obj,feature_name)).plot(kind='box')

        plt.title('Outliers fixed')

        plt.show()

        

def val_iqr_limits(df_name,col_name,w_width=None):

    """

    Description: This function is created for calculating the upper and lower limits using Tuky's IQR method.

    

    Input parameters: It accepts below two input parameters:

        1. df_name: DataFrame

        2. col_name: Feature name

        3. w_width: Whisker width provided by user and by default 1.5 

        

    Return: It returns the median, upper and lower limits of the feature based on Tuky's IQR method.

    """

    if w_width == None:

        w_width = 1.5

    else:

        w_width = w_width

        

    val_median = df_name[col_name].median()

    q1 = df_name[col_name].quantile(0.25)

    q3 = df_name[col_name].quantile(0.75)

    iqr = q3 - q1

    lower_limit = q1 - (w_width*iqr)

    upper_limit = q3 + (w_width*iqr)

#     print(val_median,q1,q3,iqr,lower_limit,upper_limit)     ## Uncomment if you want to see the values of median, q1, q2, iqr, lower and upper limit 

    return val_median, upper_limit, lower_limit



def fix_outliers(df_name,col_name,whis_width=None):

    """

    Description: This function is created for applyng the Tuky's IQR method on variable.

    

    Input parameters: It accepts the below two parameters:

        1. df_name: DataFrame

        2. col_name: Feature name

        3. whis_width: Whisker width provided by user and by default 1.5 

    

    Return: It returns the modified feature with the removed outliers.

    """

    print("######## Applied Tuky IQR Method-I ########")

    v_median, upr_limit , low_limit = val_iqr_limits(df_name,col_name,whis_width)

#     df_name[col_name] = df_name[col_name].apply(lambda val: low_limit + (val-upr_limit) if val > upr_limit 

#                                                 else upr_limit - (low_limit-val) if val < low_limit else val)

    df_name[col_name] = df_name[col_name].apply(lambda val: np.log1p(upr_limit) if val > upr_limit else np.sqrt(np.square(low_limit)) if val < low_limit else val)



    print("######## Applied Tuky IQR Method-II ########\n")

    v1_median, upr_limit1, low_limit1 = val_iqr_limits(df_name,col_name,whis_width)

    

#     df_name[col_name] = df_name[col_name].apply(lambda val: upr_limit1 if val > upr_limit1 else low_limit1 if val < low_limit1 else val)

#     df_name[col_name] = df_name[col_name].apply(lambda val: low_limit1 + (val-upr_limit1) if val > upr_limit1 

#                                                 else upr_limit1 - (low_limit1-val) if val < low_limit1 else val)

    df_name[col_name] = df_name[col_name].apply(lambda val: np.log1p(upr_limit1) if val > upr_limit1 else np.sqrt(np.square(low_limit1)) if val < low_limit1 else val)

    return df_name[col_name]



def plot_data(df_name):

    """

    This function is plotting the box plot of the dataframe.



    Parameters

    ----------

    df_name : DataFrame

        DESCRIPTION.



    Returns

    -------

    None.



    """

    with plt.style.context('seaborn'):

        plt.figure(figsize=(18,8))

        sns.boxplot(data=df_name.iloc[:,:])

        plt.title("Box-Plot Post Outliers Removal",fontdict={'size':12,'color':'blue','style':'oblique','family':'calibri'})

        plt.xticks(size=12,rotation=90,style='oblique',color='coral')

    plt.show()

    

    with plt.style.context('seaborn'):

        plt.figure(figsize=(16,8))

        sns.heatmap(df_name.corr(),cmap='coolwarm',annot=True,cbar=True,linecolor='k',linewidths=0.9)

        plt.title("Heatmap Post Outliers Removal",fontdict={'size':12,'color':'blue','style':'oblique','family':'calibri'})

        plt.xticks(size=12,rotation=90,style='oblique',color='coral')

    plt.show()
inspect_feature(patients_df_upsample,'Age')
inspect_feature(patients_df_upsample,'Total_Bilirubin')
inspect_feature(patients_df_upsample,'Direct_Bilirubin')
inspect_feature(patients_df_upsample,'Unconjugated_bilirubin')
inspect_feature(patients_df_upsample,'Alamine_Aminotransferase')
inspect_feature(patients_df_upsample,'Aspartate_Aminotransferase')
inspect_feature(patients_df_upsample,'AST_ALT_Ratio')
inspect_feature(patients_df_upsample,'Alkaline_Phosphotase')
inspect_feature(patients_df_upsample,'Total_Protiens')
inspect_feature(patients_df_upsample,'Albumin')
inspect_feature(patients_df_upsample,'Albumin_and_Globulin_Ratio')
gender = {'Female':0,'Male':1}

patients_df_upsample['Gender'] = patients_df_upsample['Gender'].apply(lambda val: gender[val])
patients_df_upsample = patients_df_upsample[['Age','Gender','Total_Bilirubin','Direct_Bilirubin','Unconjugated_bilirubin',

                                             'Alkaline_Phosphotase','Alamine_Aminotransferase','Aspartate_Aminotransferase','AST_ALT_Ratio',

                                             'Total_Protiens','Albumin','Albumin_and_Globulin_Ratio','Label']]

patients_df_upsample.head(10)
patients_df_upsample['Label'].value_counts()
patients_df_upsample.groupby(['Label','Gender']).count()['Age']
from sklearn.preprocessing import normalize
patients_df_upsample.columns
patients_df_upsample_scaled = pd.concat([pd.DataFrame(normalize(patients_df_upsample[['Age', 'Total_Bilirubin', 'Direct_Bilirubin','Unconjugated_bilirubin',

                                             'Alkaline_Phosphotase','Alamine_Aminotransferase', 'Aspartate_Aminotransferase','AST_ALT_Ratio',

                                             'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']],axis=0),columns=['Age', 'Total_Bilirubin', 'Direct_Bilirubin','Unconjugated_bilirubin',

                                             'Alkaline_Phosphotase','Alamine_Aminotransferase', 'Aspartate_Aminotransferase','AST_ALT_Ratio',

                                             'Total_Protiens', 'Albumin','Albumin_and_Globulin_Ratio']),patients_df_upsample[['Gender','Label']]],axis=1)
patients_df_upsample_scaled.head()
patients_df_upsample_scaled.shape
plot_data(patients_df_upsample_scaled.iloc[:,:-2])
patients_df_upsample_scaled.drop(['Direct_Bilirubin','Unconjugated_bilirubin','Albumin'],inplace=True,axis=1)
patients_df_upsample_scaled.to_csv('Cleaned_Trans_Scaled_Features.csv',index=False)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, train_test_split, cross_val_score, cross_validate

from sklearn.ensemble import RandomForestClassifier
def train_test_datasets(df,train_size=0.80,test_size=0.20,random_state=0):

    """

    This function is perfroming the dataset split into Train/CV/Test/Unseen datasets.

    

    Parameters

    ----------

    df : DataFrame

        DESCRIPTION.

    train_size : float, optional

        Training or Cross-Validation dataset size. The default is 0.80.

        Range --> [0.0-1.0] 

    test_size : float, optional

        Test or first model evaluation or unseen dataset size. The default is 0.20. 

        Range --> [0.0-1.0]

    random_state : int, optional

        Random state for data reproducibility. The default is 0.



    Returns

    -------

    set1_df : DataFrame 

        Training or Cross-Validation dataset.

    set2_df : DataFrame

        Test or first model evaluation or unseen dataset.

    """

    df_X = df.iloc[:,0:-1]

    df_y = df.iloc[:,-1]

    sss = StratifiedShuffleSplit(n_splits=1,train_size=train_size,test_size=test_size,random_state=random_state)

    set1_idx = []

    set2_idx = []

    for set1 , set2 in sss.split(df_X,df_y):

        set1_idx.append(set1)

        set2_idx.append(set2)

    

    set1_idx = np.array(set1_idx).flatten()

    set2_idx = np.array(set2_idx).flatten()

    

    set1_df = df.iloc[set1_idx].reset_index(drop=True)

    set2_df = df.iloc[set2_idx].reset_index(drop=True)

    return set1_df , set2_df
cv_dataset, unseen_dataset = train_test_datasets(patients_df_upsample_scaled,train_size=0.80,test_size=0.20,random_state=44)
cv_dataset.shape, unseen_dataset.shape
cvk = StratifiedShuffleSplit(n_splits=10, test_size=0.20, random_state=44)
rfc_model = RandomForestClassifier(n_estimators=25,random_state=11,

                                    max_depth=16,

                                    min_samples_split=2,

#                                     class_weight={0:0.66,1:0.37},

                                    min_samples_leaf=2,

                                    max_features='auto')
print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='f1',cv=cvk))



print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='precision',cv=cvk))

    

print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='recall',cv=cvk))
print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='f1',cv=cvk).mean())



print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='precision',cv=cvk).mean())

    

print(cross_val_score(estimator=rfc_model,X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1],scoring='recall',cv=cvk).mean())
X = cv_dataset.iloc[:,0:-1]

y = np.array(cv_dataset.iloc[:,-1])
X_train, X_test, y_train ,y_test = train_test_split(X,y,test_size=0.20,random_state=0)
rfc_model.fit(X_train, y_train)
y_pred = rfc_model.predict(X_test)
from yellowbrick.classifier import confusion_matrix as conf_matrix



## Model Metrics evaluation packages

from sklearn.metrics import f1_score, confusion_matrix, accuracy_score, precision_score, recall_score, roc_auc_score

from yellowbrick import ROCAUC
print(accuracy_score(y_test,y_pred))



print(f1_score(y_test,y_pred))



print(precision_score(y_test,y_pred))



print(recall_score(y_test,y_pred))



print(confusion_matrix(y_test,y_pred))
np.unique(y_test,return_counts=True)
unique, counts = np.unique(y_pred,return_counts=True)

unique, counts
pd.concat([pd.DataFrame(X_test.columns), pd.DataFrame(rfc_model.feature_importances_)],axis=1)
from yellowbrick.classifier import confusion_matrix as conf_matrix



visualizer = conf_matrix(rfc_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, cmap="Greens")
np.unique(y_test,return_counts=True)
unique, counts = np.unique(y_pred,return_counts=True)

unique, counts
from yellowbrick.classifier import precision_recall_curve



visualizer = precision_recall_curve(rfc_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, cmap="Greens")
roc_auc = ROCAUC(rfc_model)

roc_auc.fit(X_train,y_train)

roc_auc.score(X_test,y_test)

roc_auc.show()
import numpy as np

import pandas as pd

from sklearn.ensemble import RandomForestClassifier

import os
rfc_model = RandomForestClassifier(n_estimators=25,random_state=34,

                                    max_depth=16,

                                    min_samples_split=2,

                                    min_samples_leaf=2,

                                    max_features='auto')
pre_proc_df = pd.read_csv(os.path.join('Cleaned_Trans_Scaled_Features.csv'))
cv_dataset, unseen_dataset = train_test_datasets(pre_proc_df,train_size=0.80,test_size=0.20,random_state=91)
rfc_model.fit(cv_dataset.iloc[:,0:-1],cv_dataset.iloc[:,-1])



y_pred = rfc_model.predict(unseen_dataset.iloc[:,0:-1])

 

print(f1_score(unseen_dataset.iloc[:,-1],y_pred))



print(precision_score(unseen_dataset.iloc[:,-1],y_pred))



print(recall_score(unseen_dataset.iloc[:,-1],y_pred))



print(confusion_matrix(unseen_dataset.iloc[:,-1],y_pred))
from yellowbrick.classifier import confusion_matrix as conf_matrix



visualizer = conf_matrix(rfc_model, X_train=cv_dataset.iloc[:,0:-1],y_train=cv_dataset.iloc[:,-1], X_test=unseen_dataset.iloc[:,0:-1], y_test=unseen_dataset.iloc[:,-1], cmap="Greens")
np.unique(unseen_dataset.iloc[:,-1],return_counts=True)
unique, counts = np.unique(y_pred,return_counts=True)

unique, counts
from yellowbrick.classifier import precision_recall_curve



visualizer = precision_recall_curve(rfc_model, X_train=cv_dataset.iloc[:,0:-1],y_train=cv_dataset.iloc[:,-1], X_test=unseen_dataset.iloc[:,0:-1], y_test=unseen_dataset.iloc[:,-1], cmap="Greens")
roc_auc = ROCAUC(rfc_model)

roc_auc.fit(X=cv_dataset.iloc[:,0:-1],y=cv_dataset.iloc[:,-1])

roc_auc.score(X=unseen_dataset.iloc[:,0:-1], y=unseen_dataset.iloc[:,-1])

roc_auc.show()