import numpy as np

import pandas as pd

import os

import plotly.graph_objects as go

from plotly.subplots import make_subplots
base_dir = '/kaggle/input/kaggle-survey-2019/'

fileName = 'multiple_choice_responses.csv'

filePath = os.path.join(base_dir,fileName)

survey_df = pd.read_csv(filePath) 

responses_df = survey_df[1:]

responses_df_orig = responses_df.copy()
# Default Argument to sort by ascending or descending order

def sort_dict_by_values(dict, desc=True):

    

    # Change dict from (key,value) to (value,key) pairs

    # {'a': 20,'b': 10,'c': 30}  changes to  {20: 'a', 10: 'b', 30: 'c'}

    

    dict = {v:k    for(k,v) in dict.items()}

    

    # Sort list of tuples in Desc order

    # ([(20, 'a'), (10, 'b'), (30, 'c')])  changes to   [(30, 'c'), (20, 'a'), (10, 'b')]

    

    list_tuples = sorted(dict.items(), reverse=desc)

    

    # Create dictionary from list of tuples (Dictionary Comprehension)

    # {'a':20,'b':10,'c':30} 

    

    dict = {v:k for (k,v) in list_tuples}

    

    return dict
# Draws bar-chart from python Dictionary 

def draw_bar_chart(count_dict, title, orientation='v', color='blue'):

    

    count_series = pd.Series(count_dict)

    

    fig = go.Figure()

    

    if (orientation =='h'):

        angle = 0 

        ys= count_series.index

        xs= count_series.values

    elif (orientation =='v'):

        angle=-45

        xs= count_series.index

        ys= count_series.values

        

    trace = go.Bar(

        x=xs,

        y=ys,

        text=count_series.values,

        textposition='auto',

        marker=dict(

            color=color,

            ),

        orientation=orientation

    )



    fig.add_trace(trace)

    

    



    # Set layout properties for title, axis_tick_angle and background color

    fig.update_layout(

        xaxis_tickangle=angle,

        

        plot_bgcolor = 'White',

        

        title=dict(

            text=title,

            y=0.9,

            x=0.5,

            xanchor= 'center',

            yanchor= 'top'

        ),

        

        font=dict(

            family="Arial",

            size=14,

            color="#7f7f7f"

        ),

     )



    fig.show()
# Draws pie-chart from pandas Series (Poymorphism - Function overloading)

def draw_pie_chart(count_series, title, hole=0):

    labels = count_series.index

    sizes = count_series.values



    trace = go.Pie(labels=labels, values=sizes, hole=hole)



    layout = go.Layout(

        title=dict(

            text=title,

            y=0.9,

            x=0.5,

            xanchor= 'center',

            yanchor= 'top'

        ),

        

        font=dict(

            family="Arial",

            size=14,

            color="#7f7f7f"

        ),

    )



    data = [trace]



    fig = go.Figure(data=data, layout=layout)



    fig.show()
# Draws pie-chart from python Dictionary (Poymorphism - Function overloading)

def draw_pie_chart(count_dict, title, hole=0):

    

    count_series = pd.Series(count_dict)

    labels = count_series.index

    sizes = count_series.values



    trace = go.Pie(labels=labels, values=sizes, hole=hole)



    layout = go.Layout(

        title=dict(

            text=title,

            y=0.9,

            x=0.5,

            xanchor= 'center',

            yanchor= 'top'

        ),

        

        font=dict(

            family="Arial",

            size=14,

            color="#7f7f7f"

        ),

    )



    data = [trace]



    fig = go.Figure(data=data, layout=layout)



    fig.show()
# Create Table for Summary

def create_table(values,title='table', headerColor='blue',rowEvenColor='lightblue',rowOddColor='white',lineColor='darkslategray'):

    

    table = go.Table(

    columnorder = [1,2,3,4],

    columnwidth = [190,250,250,250],

    header = dict(

        values = [['<b>Category / Position</b>'],

                  ['<b>First</b>'],

                  ['<b>Second</b>'],

                  ['<b>Third</b>']

               ],

    

        line_color=lineColor,

        fill_color= headerColor,

        align=['left'],

        font=dict(color='black', size=13),

        height=40

      ),

  

        cells=dict(

            values=values,

            line_color='darkslategray',

        

        # 2-D list of colors for alternating rows

        fill_color = [[rowOddColor,rowEvenColor,rowOddColor, rowEvenColor,rowOddColor, rowEvenColor,rowOddColor]],

        align=['left'],

        font=dict(color='black', size=12),

        height=30)

    )

    

    fig = go.Figure(data=[table])



    # Set layout properties for title, size and background color

    fig.update_layout(

        autosize=False,

        width=900,

        height=600,

         

        plot_bgcolor = 'White',

        

        title=dict(

            text=title,

            y=0.9,

            x=0.5,

            xanchor= 'center',

            yanchor= 'top'

        ),

        

        font=dict(

            family="Arial",

            size=14,

            color="#7f7f7f"

        ),

     )



    fig.show()
responses_df = responses_df_orig.copy()



count_dict = {

    'Udacity' : (responses_df['Q13_Part_1'].count()),

    'Coursera': (responses_df['Q13_Part_2'].count()),

    'edX' : (responses_df['Q13_Part_3'].count()),

    'DataCamp' : (responses_df['Q13_Part_4'].count()),

    'DataQuest' : (responses_df['Q13_Part_5'].count()),

    'Kaggle Course' : (responses_df['Q13_Part_6'].count()),

    'Fast.ai' : (responses_df['Q13_Part_7'].count()),

    'Udemy' : (responses_df['Q13_Part_8'].count()),

    'LinkedIn Learning' : (responses_df['Q13_Part_9'].count()),

    'University Course' : (responses_df['Q13_Part_10'].count()),

    'None' : (responses_df['Q13_Part_11'].count()),

    'Other' : (responses_df['Q13_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_bar_chart(count_dict,'Platform for DataScience Courses',color='#756bb1')
responses_df = responses_df_orig.copy()



count_dict = {

    'Twitter' : (responses_df['Q12_Part_1'].count()),

    'Hacker News': (responses_df['Q12_Part_2'].count()),

    'Reddit' : (responses_df['Q12_Part_3'].count()),

    'Kaggle' : (responses_df['Q12_Part_4'].count()),

    'Course Forums' : (responses_df['Q12_Part_5'].count()),

    'YouTube' : (responses_df['Q12_Part_6'].count()),

    'Podcasts' : (responses_df['Q12_Part_7'].count()),

    'Blogs' : (responses_df['Q12_Part_8'].count()),

    'Journal Publications' : (responses_df['Q12_Part_9'].count()),

    'Slack Communities' : (responses_df['Q12_Part_10'].count()),

    'None' : (responses_df['Q12_Part_11'].count()),

    'Other' : (responses_df['Q12_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_bar_chart(count_dict,'Favorite media sources that report on data science topics',color='#fc9272')
responses_df = responses_df_orig.copy()



language_series = responses_df['Q19'].value_counts(dropna=False)



draw_pie_chart(language_series,'Programming Language Recommended for an aspiring Data Scientist',.65 )
count_dict = {

 'Jupyter' : (responses_df['Q16_Part_1'].count()),

 'RStudio': (responses_df['Q16_Part_2'].count()),

 'PyCharm' : (responses_df['Q16_Part_3'].count()),

 'Atom' : (responses_df['Q16_Part_4'].count()),

 'MATLAB' : (responses_df['Q16_Part_5'].count()),

 'Spyder' : (responses_df['Q16_Part_6'].count()),

 'Visual Studio' : (responses_df['Q16_Part_7'].count()),

 'Vim / Emacs' : (responses_df['Q16_Part_8'].count()),

 'Notepad++' : (responses_df['Q16_Part_9'].count()),

 'Sublime Text' : (responses_df['Q16_Part_10'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_bar_chart(count_dict,'Favorite media sources that report on data science topics',color='#a1d99b')
count_dict = {

 'Kaggle Notebooks' : (responses_df['Q17_Part_1'].count()),

 'Google Colab': (responses_df['Q17_Part_2'].count()),

 'Microsoft Azure Notebooks' : (responses_df['Q17_Part_3'].count()),

 'Google Cloud Notebook Products' : (responses_df['Q17_Part_4'].count()),

 'Paperspace / Gradient' : (responses_df['Q17_Part_5'].count()),

 'FloydHub ' : (responses_df['Q17_Part_6'].count()),

 'Binder / JupyterHub' : (responses_df['Q17_Part_7'].count()),

 'IBM Watson Studio' : (responses_df['Q17_Part_8'].count()),

 'Code Ocean ' : (responses_df['Q17_Part_9'].count()),

 'AWS Notebook Products' : (responses_df['Q17_Part_10'].count()),

 'None ' : (responses_df['Q17_Part_11'].count()),

 'Other ' : (responses_df['Q17_Part_12'].count()),

}





#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_bar_chart(count_dict,'Hosted Notebook Products in Regular Usage',color='#feb24c')
count_dict = {

 'Python' : (responses_df['Q18_Part_1'].count()),

 'R': (responses_df['Q18_Part_2'].count()),

 'SQL' : (responses_df['Q18_Part_3'].count()),

 'C' : (responses_df['Q18_Part_4'].count()),

 'C++' : (responses_df['Q18_Part_5'].count()),

 'Java ' : (responses_df['Q18_Part_6'].count()),

 'Javascript' : (responses_df['Q18_Part_7'].count()),

 'Typescript' : (responses_df['Q18_Part_8'].count()),

 'Bash ' : (responses_df['Q18_Part_9'].count()),

 'MATLAB' : (responses_df['Q18_Part_10'].count()),

 'None ' : (responses_df['Q18_Part_11'].count()),

 'Other' : (responses_df['Q18_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_bar_chart(count_dict,'Programming Languages in Regular Usage',color='#a6bddb')
count_dict = {

 'Ggplot' : (responses_df['Q20_Part_1'].count()),

 'Matplotlib': (responses_df['Q20_Part_2'].count()),

 'Altair' : (responses_df['Q20_Part_3'].count()),

 'Shiny' : (responses_df['Q20_Part_4'].count()),

 'D3.js' : (responses_df['Q20_Part_5'].count()),

 'Plotly' : (responses_df['Q20_Part_6'].count()),

 'Bokeh' : (responses_df['Q20_Part_7'].count()),

 'Seaborn' : (responses_df['Q20_Part_8'].count()),

 'Geoplotlib ' : (responses_df['Q20_Part_9'].count()),

 'Leaflet' : (responses_df['Q20_Part_10'].count()),

 'None ' : (responses_df['Q20_Part_11'].count()),

 'Other' : (responses_df['Q20_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_bar_chart(count_dict,'Data Visualization Tools in Regular Usage',color='#8856a7')
responses_df = responses_df_orig.copy()



count_dict = {

 'Linear or Logistic Regression' : (responses_df['Q24_Part_1'].count()),

 'Decision Trees or Random Forests': (responses_df['Q24_Part_2'].count()),

 'Gradient Boosting Machines' : (responses_df['Q24_Part_3'].count()),

 'Bayesian Approaches' : (responses_df['Q24_Part_4'].count()),

 'Evolutionary Approaches' : (responses_df['Q24_Part_5'].count()),

 'Dense Neural Networks' : (responses_df['Q24_Part_6'].count()),

 'Convolutional Neural Networks' : (responses_df['Q24_Part_7'].count()),

 'Generative Adversarial Networks ' : (responses_df['Q24_Part_8'].count()),

 'Recurrent Neural Networks' : (responses_df['Q24_Part_9'].count()),

 'Transformer Networks' : (responses_df['Q24_Part_10'].count()),

 'None' : (responses_df['Q24_Part_11'].count()),

 'Other' : (responses_df['Q24_Part_12'].count()),

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_pie_chart(count_dict,'Machine Learning Algorithms in Regular Usage',.3)
responses_df = responses_df_orig.copy()



count_dict = {

 'Automated data augmentation' : (responses_df['Q25_Part_1'].count()),

 'Automated feature engineering/selection': (responses_df['Q25_Part_2'].count()),

 'Automated model architecture searches' : (responses_df['Q25_Part_3'].count()),

 'Automated hyperparameter tuning' : (responses_df['Q25_Part_4'].count()),

 'Automation of full ML pipelines' : (responses_df['Q25_Part_5'].count()),

 'None ' : (responses_df['Q25_Part_6'].count()),

 'Other' : (responses_df['Q25_Part_7'].count()),

 }



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_pie_chart(count_dict,'Machine Learning Tools in Regular Usage',.3)
responses_df = responses_df_orig.copy()



count_dict = {

 'General purpose image/video tools' : (responses_df['Q26_Part_1'].count()),

 'Image segmentation methods': (responses_df['Q26_Part_2'].count()),

 'Object detection methods' : (responses_df['Q26_Part_3'].count()),

 'Image classification' : (responses_df['Q26_Part_4'].count()),

 'None' : (responses_df['Q26_Part_5'].count()),

 'Other' : (responses_df['Q26_Part_6'].count())

 }



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_pie_chart(count_dict,'Computer Vision Methods in Regular Usage',.3)
responses_df = responses_df_orig.copy()



count_dict = {

 'Word embeddings/vectors' : (responses_df['Q27_Part_1'].count()),

 'Encoder-decoder models': (responses_df['Q27_Part_2'].count()),

 'Contextualized embeddings' : (responses_df['Q27_Part_3'].count()),

 'Transformer language models' : (responses_df['Q27_Part_4'].count()),

 'None' : (responses_df['Q27_Part_5'].count()),

 'Other' : (responses_df['Q27_Part_6'].count())

 }



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_pie_chart(count_dict,'Natural Language Processing (NLP) Methods in Regular Usage',.3)
responses_df = responses_df_orig.copy()



count_dict = {

 'Scikit-learn' : (responses_df['Q28_Part_1'].count()),

 'TensorFlow ': (responses_df['Q28_Part_2'].count()),

 'Keras ' : (responses_df['Q28_Part_3'].count()),

 'RandomForest' : (responses_df['Q28_Part_4'].count()),

 'Xgboost' : (responses_df['Q28_Part_5'].count()),

 'PyTorch' : (responses_df['Q28_Part_6'].count()),

 'Caret': (responses_df['Q28_Part_7'].count()),

 'LightGBM  ' : (responses_df['Q28_Part_8'].count()),

 'Spark MLib' : (responses_df['Q28_Part_9'].count()),

 'Fast.ai' : (responses_df['Q28_Part_10'].count()),

 'None' : (responses_df['Q28_Part_11'].count()),

 'Other' : (responses_df['Q28_Part_12'].count()),

 }



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_pie_chart(count_dict,'Machine Learning Frameworks in Regular Usage',.3)
responses_df = responses_df_orig.copy()



count_dict = {

 'Google AutoML ' : (responses_df['Q33_Part_1'].count()),

 'H20 Driverless AI': (responses_df['Q33_Part_2'].count()),

 'Databricks AutoML' : (responses_df['Q33_Part_3'].count()),

 'DataRobot AutoML' : (responses_df['Q33_Part_4'].count()),

 'Tpot' : (responses_df['Q33_Part_5'].count()),

 'Auto-Keras' : (responses_df['Q33_Part_6'].count()),

 'Auto-Sklearn': (responses_df['Q33_Part_7'].count()),

 'Auto_ml  ' : (responses_df['Q33_Part_8'].count()),

 'Xcessiv' : (responses_df['Q33_Part_9'].count()),

 'MLbox' : (responses_df['Q33_Part_10'].count()),

 'None' : (responses_df['Q33_Part_11'].count()),

 'Other' : (responses_df['Q33_Part_12'].count()),

 }



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict)



#Draw Chart

draw_pie_chart(count_dict,'Automated Machine Learning Tools in Regular Usage',.3)
count_dict = {

    'Google Cloud Platform (GCP)' : (responses_df['Q29_Part_1'].count()),

    'Amazon Web Services (AWS)': (responses_df['Q29_Part_2'].count()),

    'Microsoft Azure' : (responses_df['Q29_Part_3'].count()),

    'IBM Cloud' : (responses_df['Q29_Part_4'].count()),

    'Alibaba Cloud' : (responses_df['Q29_Part_5'].count()),

    'Salesforce Cloud' : (responses_df['Q29_Part_6'].count()),

    'Oracle Cloud ' : (responses_df['Q29_Part_7'].count()),

    'SAP Cloud' : (responses_df['Q29_Part_8'].count()),

    'VMware Cloud' : (responses_df['Q29_Part_9'].count()),

    'Red Hat Cloud' : (responses_df['Q29_Part_10'].count()),

    'None' : (responses_df['Q29_Part_11'].count()),

    'Other' : (responses_df['Q29_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict, desc=False)



#Draw Chart

draw_bar_chart(count_dict,'Cloud Computing Platforms in Regular Usage','h',color='#c51b8a')
count_dict = {

    'AWS Elastic Compute Engine (EC2)' : (responses_df['Q30_Part_1'].count()),

    'Google Compute Engine (GCE) ': (responses_df['Q30_Part_2'].count()),

    'AWS Lambda' : (responses_df['Q30_Part_3'].count()),

    'Azure Virtual Machines' : (responses_df['Q30_Part_4'].count()),

    'Google App Engine' : (responses_df['Q30_Part_5'].count()),

    'Google Cloud Functions' : (responses_df['Q30_Part_6'].count()),

    'AWS Elastic Beanstalk ' : (responses_df['Q30_Part_7'].count()),

    'Google Kubernetes  Engine' : (responses_df['Q30_Part_8'].count()),

    'AWS Batch' : (responses_df['Q30_Part_9'].count()),

    'Azure Container Service' : (responses_df['Q30_Part_10'].count()),

    'None' : (responses_df['Q30_Part_11'].count()),

    'Other' : (responses_df['Q30_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict, desc=False)



#Draw Chart

draw_bar_chart(count_dict,'Cloud Computing Products in Regular Usage','h',color='#efedf5')
count_dict = {

    'Google BigQuery' : (responses_df['Q31_Part_1'].count()),

    'AWS Redshift': (responses_df['Q31_Part_2'].count()),

    'Databricks' : (responses_df['Q31_Part_3'].count()),

    'AWS Elastic MapReduce' : (responses_df['Q31_Part_4'].count()),

    'Teradata' : (responses_df['Q31_Part_5'].count()),

    'Microsoft Analysis Services' : (responses_df['Q31_Part_6'].count()),

    'Google Cloud Dataflow' : (responses_df['Q31_Part_7'].count()),

    'AWS Athena' : (responses_df['Q31_Part_8'].count()),

    'AWS Kinesis' : (responses_df['Q31_Part_9'].count()),

    'Google Cloud Pub/Sub' : (responses_df['Q31_Part_10'].count()),

    'None' : (responses_df['Q31_Part_11'].count()),

    'Other' : (responses_df['Q31_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict, desc=False)



#Draw Chart

draw_bar_chart(count_dict,'Big data / Analytics Products in Regular Usage','h',color='#de2d26')
count_dict = {

    'SAS' : (responses_df['Q32_Part_1'].count()),

    'Cloudera': (responses_df['Q32_Part_2'].count()),

    'Azure Machine Learning Studio' : (responses_df['Q32_Part_3'].count()),

    'Google Cloud Machine Learning Engine' : (responses_df['Q32_Part_4'].count()),

    'Google Cloud Vision' : (responses_df['Q32_Part_5'].count()),

    'Google Cloud Speech-to-Text' : (responses_df['Q32_Part_6'].count()),

    'Google Cloud Natural Language' : (responses_df['Q32_Part_7'].count()),

    'RapidMiner' : (responses_df['Q32_Part_8'].count()),

    'Google Cloud Translation' : (responses_df['Q32_Part_9'].count()),

    'Amazon SageMaker' : (responses_df['Q32_Part_10'].count()),

    'None' : (responses_df['Q32_Part_11'].count()),

    'Other' : (responses_df['Q32_Part_12'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict, desc=False)



#Draw Chart

draw_bar_chart(count_dict,'Machine Learning Products in Regular Usage','h',color='#ffeda0')
count_dict = {

    'MySQL' : (responses_df['Q34_Part_1'].count()),

    'PostgresSQL': (responses_df['Q34_Part_2'].count()),

    'SQLite' : (responses_df['Q34_Part_3'].count()),

    'Microsoft SQL Server' : (responses_df['Q34_Part_4'].count()),

    'Oracle Database' : (responses_df['Q34_Part_5'].count()),

    'Microsoft Access' : (responses_df['Q34_Part_6'].count()),

    'AWS DynamoDB' : (responses_df['Q34_Part_7'].count()),

    'Azure SQL Database' : (responses_df['Q34_Part_8'].count()),

    'Google Cloud SQL' : (responses_df['Q34_Part_9'].count()),

    'None' : (responses_df['Q34_Part_10'].count()),

    'Other' : (responses_df['Q34_Part_11'].count())

}



#Sort dictionary by values

count_dict = sort_dict_by_values(count_dict, desc=False)



#Draw Chart

draw_bar_chart(count_dict,'Relational Database Products in Regular Usage','h',color='#e34a33')
values = [['<b>Platforms</b>' ,'<b>Favorite Media sources</b>' ,'<b>Programming Language to learn first</b>',\

          '<b>Integrated Development Environments (IDEs)</b>','<b>Hosted Notebook Products</b>',\

          '<b>Programming Languages</b>','<b>Data Visualization Tools</b>'],

          ['Coursera', 'Kaggle', 'Python',  'Jupyter','NONE','Python','Matplotlib'],

          ['Kaggle', 'Blogs', 'NULL', 'Spyder','Kaggle Notebooks','SQL','Seaborn'],

          ['Udemy', 'Youtube', 'R',   'RStudio','Google Colab','R','Ggplot']

         ]



create_table(values= values, title='Learning Platform and Development Environment',\

             headerColor = '#a4a1fc',rowEvenColor = '#f1f1fe');



values = [['<b>Machine Learning Algorithms</b>','<b>Machine Learning Tools</b>',\

           '<b>Computer Vision Methods</b>','<b>Natural Language Processing Methods</b>','<b>Machine Learning Frameworks</b>',\

          '<b>Automated Machine Learning Tool</b>'],

          

          ['Linear or Logistic Regression','OTHER',\

           'Image classification & other networks(VGG, Inception, ResNet, ResNeXt, etc)',\

          'Word embeddings/vectors (GLoVe, fastText, word2vec)','Scikit-learn','NONE'],

          

          ['Decision Trees or Random Forests','Automated model architecture searches (e.g. darts, enas)',\

           'General purpose image/video tools (PIL, cv2, skimage, etc)',\

          'Encoder-decoder models (seq2seq, vanilla transformers)','TensorFlow','Auto-Sklearn'],

          

          ['Gradient Boosting Machines (xgboost, lightgbm, etc)','Automated data augmentation (e.g. imgaug, albumentations)','Image segmentation methods (U-Net, Mask R-CNN, etc)',\

          'Transformer language models (GPT-2, BERT, XLnet, etc)','Keras','Google AutoML'],

                

         ]



create_table(values= values, title='Trends in Machine Learning',\

             headerColor = '#f8a630',rowEvenColor = '#fdf6d8');



values = [['<b>Cloud Computing Platforms','<b>Cloud Computing Products<b>',\

           '<b>Big data / Analytics Products<b>','<b>Machine Learning Products<b>',\

          '<b>Relational Database Products<b>'],

          

          ['Amazon Web Services (AWS)','NONE','NONE','NONE','MySQL'],

          

          ['NONE','AWS Elastic Compute Engine (EC2)','Google BigQuery','Google Cloud Machine Learning Engine','PostgresSQL'],

          

          ['Google Cloud Platform (GCP)','Google Cloud Platform (GCP)','Databricks',\

          'Azure Machine Learning Studio','Microsoft SQL Server'],

                 

         ]

create_table(values= values, \

             title='Trends in Products',\

             headerColor = '#e87efb',rowEvenColor = '#f8d8fd');
