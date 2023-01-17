import pandas as pd 

import numpy as np 

import matplotlib.pyplot as plt

import seaborn as sb

import warnings

%matplotlib inline
#load data

df= pd.read_csv("../input/students-performance-in-exams/StudentsPerformance.csv")
# high-level overview of data shape and composition

print(df.shape)

print(df.head(10))

print(df.info())
#copy from orginal df

df2=df.copy() 

df2.head()
#renaming the columns

df2.rename(columns = {"race/ethnicity": "race", 

                           "parental level of education":"parent_education", 

                           "test preparation course": "test_preparation_course",

                            "math score":"maths_score",

                            "reading score":"reading_score",

                            "writing score":"writing_score"},inplace=True)

df2.head()
# creating a new columns pass_math,reading_math,writing_math this columns will tell us whether the students are pass or fail

passmarks = 40  #assume passmark=40

df2['pass_math'] = np.where(df2['maths_score']< passmarks, 'Fail', 'Pass')

df2['pass_reading'] = np.where(df2['reading_score']< passmarks, 'Fail', 'Pass')

df2['pass_writing'] = np.where(df2['writing_score']< passmarks, 'Fail', 'Pass')
#creating a new columns of total score of each student

df2['Total_score']= df2['maths_score']+df2['writing_score']+df2['reading_score']
#creating a new column of percentage of each student

df2['percentage_score']=df2['Total_score']/300
#creating a new columns pass/fail result of each student

def result(percentage_score):

    if percentage_score >=0.33:

        return "Pass"

    else:

        return "Fail"

    

df2['result_score']=df2['percentage_score'].apply(result)
#the new data

df2.head()
# descriptive statistics for numeric variables

df2.describe().T
df2["parent_education"].value_counts().index,df2["race"].value_counts().index

# convert parental level of education and race/ethnicity into ordered categorical types

ordinal_var_dict = {'parent_education': ["some high school","high school","some college","associate's degree"

                                                    ,"bachelor's degree","master's degree"],

                    'race': ['group E', 'group D', 'group C', 'group B', 'group A']}



for var in ordinal_var_dict:

    ordered_var = pd.api.types.CategoricalDtype(ordered = True,

                                                categories = ordinal_var_dict[var])

    df2[var] = df2[var].astype(ordered_var)
# create the plot

default_color = sb.color_palette('viridis',9)[3]

plt.figure(figsize = [8, 6])

sb.countplot(data = df2, x = 'parent_education', color = default_color);

# add annotations

n_df=df2.shape[0]

parental_counts = df2['parent_education'].value_counts()

locs, labels = plt.xticks() 

# loop through each pair of locations and labels

for loc, label in zip(locs, labels):



    # get the text property for the label to get the correct count

    count = parental_counts[label.get_text()]

    pct_string = '{:0.1f}%'.format(100*count/n_df)



    # print the annotation just below the top of the bar

    plt.text(loc, count-8, pct_string, ha = 'center', color = 'w',fontweight = 30)

    

plt.xticks(rotation=90);

plt.title('Comparison of parent education level');

plt.yticks([0,25,50,75,100,125,150,175,200,225]);

plt.xlabel("Education levels");

# create the plot

default_color = sb.color_palette('viridis',9)[3]

plt.figure(figsize = [8, 6])

sb.countplot(data = df2, x = 'race', color = default_color);

# add annotations

parental_counts = df2['race'].value_counts()

locs, labels = plt.xticks() 

# loop through each pair of locations and labels

for loc, label in zip(locs, labels):



    # get the text property for the label to get the correct count

    count = parental_counts[label.get_text()]

    pct_string = '{:0.1f}%'.format(100*count/n_df)



    # print the annotation just below the top of the bar

    plt.text(loc, count-8, pct_string, ha = 'center', color = 'w')

    

plt.xticks(rotation=90);

plt.title('Comparison of race Groups', fontweight = 30);

plt.yticks([0,50,100,150,200,250,300,350]);

plt.xlabel("Groups");

plt.figure(figsize = [8, 6])

colors=( 'mistyrose', 'skyblue')

plt.pie(df2['gender'].value_counts()/n_df,labels=('Female','Male'),

        explode = [0.08,0.08],autopct ='%1.1f%%'

        ,shadow = True,startangle = 90, textprops={'fontsize': 12},colors=colors);

plt.title('Students Number of Males and Femals  ', fontsize=12 );

plt.axis('equal');

plt.ylabel('count')



plt.show()

plt.figure(figsize = [8, 6])

colors=( 'silver', 'teal')

plt.pie(df2['lunch'].value_counts()/n_df,labels=('standard','free/reduced'),

        explode = [0.08,0.08],autopct ='%1.1f%%'

        ,shadow = True,startangle = 90, textprops={'fontsize': 12},colors=colors)

plt.title('Comparison of different types of lunch', fontsize=12 );

plt.axis('equal')

plt.ylabel('count')

plt.show()

plt.figure(figsize = [8, 6])

colors=( 'silver', 'teal')

plt.pie(df2['test_preparation_course'].value_counts()/n_df,labels=('none','completed'),

        explode = [0.08,0.08],autopct ='%1.1f%%'

        ,shadow = True,startangle = 90, textprops={'fontsize': 12},colors=colors)

plt.title('test preparation course(none/completed)) ', fontsize=12 );

plt.axis('equal')

plt.ylabel('count')

plt.show()

plt.figure(figsize = [8, 6])

plt.pie(df2['pass_math'].value_counts()/n_df,labels=('Pass','Fail'),

        explode = [0.08,0.08],autopct ='%1.1f%%'

        ,shadow = True,startangle = 45, textprops={'fontsize': 12},colors=colors)

plt.title('Pass/Fail in Maths', fontweight = 12, fontsize = 12)

plt.ylabel('count')

plt.axis('equal')

plt.show()
plt.figure(figsize = [8, 6])

plt.pie(df2['pass_reading'].value_counts()/n_df,labels=('Pass','Fail'),

        explode = [0.08,0.08],autopct ='%1.1f%%'

        ,shadow = True,startangle = 45, textprops={'fontsize': 12},colors=colors)

plt.title('Pass/Fail in Reading', fontweight = 12, fontsize = 12)

plt.ylabel('count')

plt.axis('equal')

plt.show()
plt.figure(figsize = [8, 6])

plt.pie(df2['pass_writing'].value_counts()/n_df,labels=('Pass','Fail'),

        explode = [0.08,0.08],autopct ='%1.1f%%'

        ,shadow = True,startangle = 45, textprops={'fontsize': 12},colors=colors)

plt.title('Pass/Fail in writing', fontweight = 12, fontsize = 12)

plt.ylabel('count')

plt.axis('equal')

plt.show()
plt.figure(figsize = [8, 6])

plt.pie(df2['result_score'].value_counts()/n_df,labels=('Pass','Fail'),

        explode = [0.08,0.08],autopct ='%1.1f%%'

        ,shadow = True,startangle = 45, textprops={'fontsize': 12},colors=colors)

plt.title('Pass/Fail Total Score of students ', fontweight = 12, fontsize = 12)

plt.ylabel('count')

plt.axis('equal')

plt.show()
# univariate plot of math score

plt.figure(figsize = [8, 6])

d_bins=np.arange(0, df2['maths_score'].max()+5, 5)

sb.distplot(df2["maths_score"],d_bins);

plt.yticks([0, 0.005,0.010, 0.015, 0.020,0.025,0.030],[0, "0.5%","1%"," 1.5%","2%","2.5%","3%"]);

plt.xticks([0, 10,20,30,40,50,60,70,80,90,100]);

plt.title(" maths score distribution of students");

plt.ylabel("proportion of students");
# univariate plot of reading score

plt.figure(figsize = [8, 6])

d_bins=np.arange(0, df2['reading_score'].max()+5, 5)

sb.distplot(df2["reading_score"],d_bins);

plt.yticks([0, 0.005,0.010, 0.015, 0.020,0.025,0.030],[0, "0.5%","1%"," 1.5%","2%","2.5%","3%"]);

plt.xticks([0, 10,20,30,40,50,60,70,80,90,100]);

plt.title(" reading score distribution of students");

plt.ylabel("proportion of students");
# univariate plot of writing score

plt.figure(figsize = [8, 6])

d_bins=np.arange(0, df2['writing_score'].max()+5, 5)

sb.distplot(df2["writing_score"],d_bins);

plt.yticks([0, 0.005,0.010, 0.015, 0.020,0.025,0.030],[0, "0.5%","1%"," 1.5%","2%","2.5%","3%"]);

plt.xticks([0, 10,20,30,40,50,60,70,80,90,100]);

plt.title(" writing score distribution of students");

plt.ylabel("proportion of students");
# univariate plot of Total score

plt.figure(figsize = [8, 6])

d_bins=np.arange(0, df2['Total_score'].max()+8, 8)

sb.distplot(df2["Total_score"],d_bins);

plt.yticks([0, 0.002,0.004, 0.006, 0.008,0.010,0.012],[0, "0.2%","0.4%"," 0.6%","0.8%","1%","1.2%"]);

plt.title(" Total score  distribution of students");

plt.ylabel("proportion of students");
# univariate plot of math score

plt.figure(figsize = [8, 6])

d_bins=np.arange(0, df2['percentage_score'].max()+0.05, 0.05)

sb.distplot(df2["percentage_score"],d_bins);

plt.title(" percentage score  distribution of students");

plt.ylabel("proportion of students");
#list of categoric features

categoric_vars = ['gender', 'race', 'parent_education', 'lunch', 'test_preparation_course', 'pass_math', 'pass_reading',"pass_writing","result_score"]

#list of numeric features

numeric_vars = ['maths_score', 'reading_score', 'writing_score',"Total_score","percentage_score"]

# correlation plot

plt.figure(figsize = [8, 6])

sb.heatmap(df2[numeric_vars].corr(), annot = True, fmt = '.3f',

           cmap = 'terrain', center = 0)

plt.show()
g = sb.PairGrid(data = df2, vars = numeric_vars)

g = g.map_diag(plt.hist, bins = 20);

g.map_offdiag(plt.scatter);
# plot matrix of numeric features against categorical features.

# can use a larger sample since there are fewer plots and they're simpler in nature.

warnings.filterwarnings("ignore")



samples = np.random.choice(df2.shape[0], 400, replace = False)

students_samp = df2.loc[samples,:]



def boxgrid(x, y, **kwargs):

    """ Quick hack for creating box plots with seaborn's PairGrid. """

    default_color = sb.color_palette()[0];

    sb.boxplot(x, y, color = default_color);

    plt.xticks(rotation=15);



plt.figure(figsize = [10, 10]);

g = sb.PairGrid(data = students_samp, y_vars = ['Total_score', 'percentage_score'], x_vars = categoric_vars,

               size = 3, aspect = 1.5);

g.map(boxgrid);

plt.show();
#relation between gender and marks scored in each subject

plt.figure(figsize=(14,5))

plt.subplot(1, 3, 1);

default_color = sb.color_palette('vlag',9)[5]



   #plot for maths score

sb.boxplot(data=df2,x='gender', y='maths_score',color=default_color)

plt.title("Maths Score  .VS.  gender");



    #plot for reading score

plt.subplot(1, 3, 2);

sb.boxplot(data=df2,x='gender', y='reading_score',color=default_color)

plt.title("Reading Score .VS. gender");

    

    #plot for writing score

plt.subplot(1, 3, 3);

sb.boxplot(data=df2,x='gender', y='writing_score',color=default_color)

plt.title("writing score .VS. gender");

    

plt.show();
#relation between gender and marks scored in each subject

warnings.filterwarnings("ignore")

df2.groupby('gender')['maths_score',"reading_score","writing_score"].mean().T

#some calculations

print(df2.groupby('gender')["pass_math"].value_counts())

print(df2.groupby('gender')["pass_reading"].value_counts())

print(df2.groupby('gender')["pass_writing"].value_counts())
plt.figure(figsize=(20,5))

plt.subplot(1, 3, 1);

default_color = sb.color_palette('vlag',9)[8]



   #plot for maths score

sb.countplot(data = df2, x = 'gender', hue = 'pass_math',palette = 'Greens')

plt.title("Maths Pass/Fail  .VS.  gender");



    #plot for reading score

plt.subplot(1, 3, 2);

sb.countplot(data = df2, x = 'gender', hue = 'pass_reading',palette = 'Greens')

plt.title("Reading Pass/Fail .VS. gender");

    

    #plot for writing score

plt.subplot(1, 3, 3);

sb.countplot(data = df2, x = 'gender', hue = 'pass_writing',palette = 'Greens')

plt.title("writing Pass/Fail .VS. gender");   

plt.show();

#relations between gender and total,percentage score 



plt.figure(figsize=(14,5))

plt.subplot(1, 2, 1);

default_color = sb.color_palette('vlag',9)[5]



   #plot for gender & total score score

sb.boxplot(data=df2,x='gender', y='Total_score',color=default_color)

plt.title("Total Score  .VS.  gender");



   #plot for gender & percentage score 

plt.subplot(1, 2, 2);

sb.boxplot(data=df2,x='gender', y='percentage_score',color=default_color)

plt.title("percentage Score .VS. gender");

    

plt.show();
df2.groupby('gender')['Total_score',"percentage_score"].mean().T

print(df2.groupby('gender')["result_score"].value_counts())
#plot for result score and gender 

plt.figure(figsize=(8,5))

default_color = sb.color_palette('vlag',9)[8]

sb.countplot(data = df2, x = 'gender', hue = 'result_score'

,palette = 'Greens')

plt.title("Result score  .VS.  gender");

#relations between gender and race,lunch,test_preparation_course





plt.figure(figsize=(14,5))

plt.subplot(1, 3, 1);

default_color = sb.color_palette('vlag',9)[8]



   #plot for race

sb.countplot(data = df2, x = 'race', hue = 'gender',palette = 'Blues')

plt.title("Race .VS.  gender");



    #plot for lunch

plt.subplot(1, 3, 2);

sb.countplot(data = df2, x = 'lunch', hue = 'gender',palette = 'Blues')

plt.title("Lunch .VS. gender");

    

    #plot for test_preparation_course

plt.subplot(1, 3, 3);

sb.countplot(data = df2, x = 'test_preparation_course', hue = 'gender',palette = 'Blues')

plt.title("Test preparation course .VS. gender");   

plt.show();
#relations between parent education and maths score,reading score,writing score,Total score



p1=df2.groupby('parent_education')['maths_score'].mean().sort_values()

p2=df2.groupby('parent_education')['reading_score'].mean().sort_values()

p3=df2.groupby('parent_education')['writing_score'].mean().sort_values()

p4=df2.groupby('parent_education')['Total_score'].mean().sort_values()



color=sb.color_palette()[0]

plt.figure(figsize=(20,5))

plt.subplot(1, 4, 1);    

sb.pointplot(data=df2,x=p1.index, y=p1,color=color)

plt.title("Maths Score avarge VS parent education");

plt.xticks(rotation=90);





plt.subplot(1, 4, 2);    

sb.pointplot(data=df2,x=p2.index, y=p2,color=color)

plt.title("Reading Score avarge VS parent education");

plt.xticks(rotation=90);





plt.subplot(1, 4, 3);    

sb.pointplot(data=df2,x=p3.index, y=p3,color=color)

plt.title("Writing Score avarge VS parent education");

plt.xticks(rotation=90);





plt.subplot(1, 4, 4);    

sb.pointplot(data=df2,x=p4.index, y=p4,color=color)

plt.title("Total Score avarge VS parent education");

plt.xticks(rotation=90);

#relations between Parent education and gender , Result score,Test preparation course





plt.figure(figsize=(14,5))

default_color = sb.color_palette('vlag',9)[8]



plt.subplot(1, 3, 1);    

sb.countplot(data = df2, x = 'parent_education', hue = 'gender',palette = 'Blues')

plt.xticks(rotation=90);

plt.title("Parent education .VS.  gender");



plt.subplot(1, 3,2);

sb.countplot(data = df2, x = 'parent_education', hue = 'result_score',palette = 'Blues')

plt.legend(loc = 6, ncol = 3, framealpha = 1, title = 'Result score')



plt.xticks(rotation=90);



plt.title("Parent education .VS. Result score");   



plt.subplot(1, 3,3);

sb.countplot(data = df2, x = 'parent_education', hue = 'test_preparation_course',palette = 'Blues')

plt.legend(loc = 8, ncol = 3, framealpha = 1, title = 'Test preparation course')

plt.xticks(rotation=90);



plt.title("Parent education .VS. Test preparation course");   

plt.show();
#relations between Race and maths score,reading score,writing score,Total score





g1=df2.groupby('race')['maths_score'].mean().sort_values()

g2=df2.groupby('race')['reading_score'].mean().sort_values()

g3=df2.groupby('race')['writing_score'].mean().sort_values()

g4=df2.groupby('race')['Total_score'].mean().sort_values()



color=sb.color_palette()[0]

plt.figure(figsize=(18,5))

plt.subplot(1, 4, 1);    

sb.pointplot(data=df2,x=g1.index, y=g1,color=color)

plt.title("Maths Score avarge  VS race");

plt.xticks(rotation=15);





plt.subplot(1, 4, 2);    

sb.pointplot(data=df2,x=g2.index, y=g2,color=color)

plt.title("Reading Score avarge VS race");

plt.xticks(rotation=15);





plt.subplot(1, 4, 3);    

sb.pointplot(data=df2,x=g3.index, y=g3,color=color)

plt.title("Writing Score avarge VS race");

plt.xticks(rotation=15);





plt.subplot(1, 4, 4);    

sb.pointplot(data=df2,x=g4.index, y=g4,color=color)

plt.title("Total Score avarge  VS race");

plt.xticks(rotation=15);



#relations between Race and result score,Test preparation course

plt.figure(figsize=(14,5))

default_color = sb.color_palette('vlag',9)[8]



plt.subplot(1, 2, 1);

sb.countplot(data = df2, x = 'race', hue = 'result_score',palette = 'Blues')

plt.title("Race .VS. Result score");



plt.subplot(1, 2, 2);

sb.countplot(data = df2, x = 'race', hue = 'test_preparation_course',palette = 'Blues')

plt.title("Race .VS. Test preparation course");

    
#relation between Race and Parent education

plt.figure(figsize=(14,5))

sb.countplot(data = df2, x = 'race', hue = 'parent_education',palette = 'Blues');

plt.title("Race .VS. Parent education");

plt.xticks(rotation=15);
#relations between Total score and maths,reading,writing score





plt.figure(figsize = [15, 6])



plt.subplot(1, 3, 1);

plt.scatter(data = df2, x = 'maths_score', y = 'Total_score', alpha = 1/6)

plt.xlabel('maths score')

plt.ylabel('total score ')

plt.title("maths score .VS. total score ")



plt.subplot(1, 3, 2);

plt.scatter(data = df2, x = 'reading_score', y = 'Total_score', alpha = 1/6)

plt.xlabel('reading score')

plt.ylabel('total score ')

plt.title("reading score .VS.  total score ")



plt.subplot(1, 3, 3);

plt.scatter(data = df2, x = 'writing_score', y = 'Total_score', alpha = 1/6)

plt.xlabel('writing score')

plt.ylabel('total score ')

plt.title("writing score .VS.  total score ") 



plt.show()
# create faceted scatter plots  on levels of race and parent_education

warnings.filterwarnings("ignore")





g = sb.FacetGrid(data = df2, col = 'race', row = 'parent_education', size = 2.5,

                margin_titles = True)

g.map(plt.scatter, 'Total_score', 'percentage_score');

#Multivariate relations between 

#result score vs (parent education & Total score),(race & Total score),(test preparation course & Total score)



plt.figure(figsize=[20,5])



plt.subplot(1,3,1)

sb.pointplot(data = df2, x = 'parent_education', y = 'Total_score', hue = 'result_score',

                  dodge = 0.3, linestyles = "")

plt.xticks(rotation=90);

plt.legend(loc = 5, title = 'result_score')

plt.title("parent education & Total score .VS. result score ")







plt.subplot(1,3,2)

sb.pointplot(data = df2, x = 'race', y = 'Total_score', hue = 'result_score',

                  dodge = 0.3, linestyles = "")

plt.xticks(rotation=90);

plt.title("race & Total score .VS. result score ")





plt.subplot(1,3,3)

sb.pointplot(data = df2, x = 'test_preparation_course', y = 'Total_score', hue = 'result_score',

                  dodge = 0.3, linestyles = "")

plt.xticks(rotation=90);

plt.legend(loc = 5, title = 'result_score')

plt.title("test preparation course & Total score .VS. result score ");

#Multivariate relations between 

#(gender & race) vs (maths score),(reading score),(writing score)



plt.figure(figsize=[20,5])



plt.subplot(1,4,1)

sb.pointplot(data = df2, x = 'gender', y = 'maths_score', hue = 'race',

                  dodge = 0.3, linestyles = "")

plt.xticks(rotation=90);

plt.title("gender & race .VS. maths score ")







plt.subplot(1,4,2)

sb.pointplot(data = df2, x = 'gender', y = 'reading_score', hue = 'race',

                  dodge = 0.3, linestyles = "")

plt.xticks(rotation=90);

plt.title("gender & race .VS. reading score ")





plt.subplot(1,4,3)

sb.pointplot(data = df2, x = 'gender', y = 'writing_score', hue = 'race',

                  dodge = 0.3, linestyles = "")

plt.xticks(rotation=90);

plt.title("gender & race .VS. writing score ");





plt.subplot(1,4,4)

sb.pointplot(data = df2, x = 'gender', y = 'Total_score', hue = 'race',

                  dodge = 0.3, linestyles = "")

plt.xticks(rotation=90);

plt.title("gender & race .VS. result score ");
#relations between 

#(result score & percentage) vs (race)





plt.figure(figsize=[14,5])



sb.barplot(data=df2,x='result_score',y='percentage_score',hue='race',palette = 'Greens');

plt.title("result score & percentage score Vs Race");

#relations between 

#(result score & percentage) vs (Parent education)

plt.figure(figsize=[14,5])



sb.barplot(data=df2,x='result_score',y='percentage_score',hue='parent_education',palette = 'Greens');

plt.title("result score & percentage score Vs Parent education");

#relations between 

#(result score & percentage) vs (Gender)



plt.figure(figsize=[14,5])



sb.barplot(data=df2,x='result_score',y='percentage_score',hue='gender',palette = 'Blues');

plt.title("result score & percentage score Vs Gender");
