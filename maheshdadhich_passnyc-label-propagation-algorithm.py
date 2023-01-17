# Loading Packages 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly 
import time
import os 
import folium 
import scipy.stats
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import KMeans
from scipy.spatial import distance
from plotly.offline import init_notebook_mode, iplot
init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.figure_factory as ff
from sklearn import preprocessing
from plotly import tools
import plotly.plotly as py
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import sklearn.semi_supervised 
from sklearn import datasets
from sklearn.semi_supervised import LabelPropagation
# importing files 
dsfg_path = '../input/data-science-for-good/'
safety_path = '../input/ny-2010-2016-school-safety-report/'
class_size_path = '../input/ny-2010-2011-class-size-school-level-detail/'
# Reading school_df 
schools_df = pd.read_csv(dsfg_path+'2016 School Explorer.csv')
# Sanity check 
print("Shape of Schools data is {}".format(schools_df.shape))
print("Number of nulls in data are {}".format(schools_df.isnull().sum().sum()))
schools_df.head()
# loading class size data
class_size = pd.read_csv(class_size_path + '2010-2011-class-size-school-level-detail.csv')
class_size.head()
# Laoding district 5 SAT data
SAT_df = pd.read_csv(dsfg_path+ 'D5 SHSAT Registrations and Testers.csv')
print(SAT_df.shape)
SAT_df.head()
# Function for treating Outliers 
# 1 outlier treatment
def outlier_treatment(df,columns):
    """Function to cap outliers"""
    temp=pd.DataFrame(columns=['variable','q75','q25'],index=range(len(columns)))
    i=0    
    for col in columns:
        temp['variable'][i]=col
        q75, q25 = np.nanpercentile(df[col], [75 ,25])
        caps1,caps2=np.nanpercentile(df[col],[5,95])
        h= (q75 - q25)*1.5
        temp['q75'][i]=q75+h
        temp['q25'][i]=q25-h
        df[col]=np.where(df[col]>(q75+h),caps2,df[col])
        df[col]=np.where(df[col]<(q25-h),caps1,df[col])
        i=i+1
    return df

# 2 Function converting string % values to int
def percent_to_int(df_in):
    """Function to make % strings cols to float
        credit - infocusp's script """
    for col in df_in.columns.values:
        if col.startswith("Percent") or col.endswith("%") or col.endswith("Rate"):
            df_in[col] = df_in[col].astype(np.object).str.replace('%', '').astype(float)
    return(df_in)

# 3 Label for categories 
def Label_for_cat_var(df, col):
    """Function to define labels for categorical columns"""
    le = preprocessing.LabelEncoder()
    le.fit(df[col])
    df[col] = le.transform(df[col])
    del le
    return(df)

# 4 #Binning:
def binning(col, cut_points, labels=None):
    """Function to assign bins for a given variable"""
    #Define min and max values
    minval = col.min()
    maxval = col.max()

    #create list by adding min and max to cut_points
    break_points =  [0.0]+ cut_points + [100.0]

    #if no labels provided, use default labels 0 ... (n-1)
    if not labels:
        labels = range(len(cut_points)+1)

    #Binning using cut function of pandas
    colBin = pd.cut(col,bins=break_points,labels=labels,include_lowest=True)
    return(colBin)

# 5 Crime to perc 
def perc_crime_transformation(df):
    """Function to convert crime to a percentage value"""
    for col in tranform_list:
        max_col = df[col].max()
        min_col = df[col].min()
        delta = max_col - min_col
        for i in list(range(crime_affect.shape[0])):
            df.loc[i, col] = (df.loc[i, col]- min_col)*100.0/(delta)
    return(df)

# 6. Deciding clusters for location, 32 district is much more, we don't need that much variations
def elbow_curve(df):
    """function to determine the number of cluster using elbow curve"""
    columns = ['Latitude','Longitude']
    df_new = df[columns]
    Nc = range(1, 50)
    kmeans = [KMeans(n_clusters=i) for i in Nc]
    kmeans
    score = [kmeans[i].fit(df_new).score(df_new) for i in range(len(kmeans))]
    score
    plt.plot(Nc,score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()
    return(0)

# 7. Assign clusters 
def Assign_k_means_cluster(df, n):
    """function to assign k-means clusters 
        n = number of clusters
    """
    columns = ['Latitude','Longitude']
    df_new = df[columns]
    kmeans = MiniBatchKMeans(n_clusters=n).fit(df_new)
    df.loc[:, 'location_cluster'] = kmeans.predict(df[columns])
    #test_meta.loc[:, 'k_means_cluster'] = kmeans.predict(test_meta[columns])
    return(df)

# 8. Function to convert a string to upper case
def to_upper(row):
    """Function to convert a string to upper case """
    return(row.upper())
# creating master_df 
# converting all perc values in school_df to int values 
schools_df = percent_to_int(schools_df)
# Removing outliers from the dataframe 
cols_to_filter_outlier = []
subj= {1:'Math',
       2:'ELA'}
grade_name = [3,4,5,6,7,8]
for grade in grade_name:
    for p in [1,2]:
        var = 'Grade '+ str(grade)  +' '+subj[p]+' 4s - All Students'
        cols_to_filter_outlier.append(var)
        
print(cols_to_filter_outlier)
schools_df = outlier_treatment(schools_df,cols_to_filter_outlier)
schools_df.head()
# finding optimum number of location clusters in schools_df
elbow_curve(schools_df) # To check what is the optimum number of clusters 
# Assigning locations clusters in schools_df 
# I name it bayes df as same df will be used in bayesian analysis
bayes_df = Assign_k_means_cluster(schools_df, 10)
bayes_df.head()
# Assigning perc of 4s in different classes 
# columns which are already given in percentage values 
cols_having_perc = []
for col in bayes_df.columns.values:
    if col.startswith("Percent") or col.endswith("%") or col.endswith("Rate"):
        cols_having_perc.append(col)
cols_having_perc # percentage of columns having % in them 



# Columns we create having perc
# Converting all grades values to perc 
grades_name = [6,7,8] # 3,4,5,
bayes_df.rename(columns = {'Grade 3 Math - All Students tested':'Grade 3 Math - All Students Tested'}, inplace = True)



#assigning a indicator class_size_zero = 0 or 1
print("size before ", bayes_df.shape)



# It doesn't take it if all students in that class are zero 
for grade in grades_name:
    for subj in ['ELA', 'Math']:
        All = 'Grade '+str(grade)+' '+subj+' - All Students Tested'
        bayes_df = bayes_df.loc[bayes_df[All]!=0]


        
        
print(bayes_df.shape)
created_perc_col = []
for grade in grades_name:
    for subj in ['ELA', 'Math']:
        col_name = subj+'_'+str(grade)+'_4s'
        created_perc_col.append(col_name)
        All_4s = 'Grade '+str(grade)+' '+subj+' 4s - All Students'
        All = 'Grade '+str(grade)+' '+subj+' - All Students Tested'
        bayes_df.loc[:,col_name] = (bayes_df[All_4s].values*100.0)/bayes_df[All].values
        

# Assiging label for non-numeric columns 
non_numeric_cols = ['Rigorous Instruction Rating','Collaborative Teachers Rating','Supportive Environment Rating','Effective School Leadership Rating','Strong Family-Community Ties Rating','Trust Rating','Student Achievement Rating']
bayes_df.dropna(axis =0, subset =non_numeric_cols,  inplace = True)
print(bayes_df.shape)
for col in non_numeric_cols:
    bayes_df = Label_for_cat_var(bayes_df, col)
# Rolling up the time series data 
SAT_df.head()
aggregate_func = {'Enrollment on 10/31':np.mean,
                  'Number of students who registered for the SHSAT':np.mean,
                  'Number of students who took the SHSAT':np.mean}

print("Shape of SAT d5 data is {}".format(SAT_df.shape))
SAT_summary = pd.DataFrame(SAT_df.groupby(['DBN','School name']).agg(aggregate_func))
print("Shape of SAT summary is {}".format(SAT_summary.shape))
SAT_summary.reset_index(inplace = True)
SAT_summary.head()



# SAT_summary Designing features for getting under performance - performance indicators
SAT_summary['Registration_rate'] = SAT_summary['Number of students who registered for the SHSAT']/(SAT_summary['Enrollment on 10/31'])
SAT_summary['Appear_rate'] = SAT_summary['Number of students who took the SHSAT']/(SAT_summary['Number of students who registered for the SHSAT'])



# Low performing school
# Any Schools whose registration_rate and Apprear rate will be less 1 std below mean values are under performing
mean_reg_rate = SAT_summary['Registration_rate'].mean()
mean_App_rate = SAT_summary['Appear_rate'].mean()
std_reg_rate  = SAT_summary['Registration_rate'].std()
std_App_rate  = SAT_summary['Appear_rate'].std()



# Threshold decision - 
# **Basically use complete data to get the mean and thresholds and not do it based on D5 data**
n = 1.0  # anything which is one standard deviation below mean is underperforming

threshold_reg = mean_reg_rate - n*std_reg_rate
threshold_App = mean_App_rate - n*std_App_rate

def Assign_underperforming(row):
    """Function to Assign if a school is under performing or not"""
    underperform = 0
    if row['Registration_rate'] <= threshold_reg:
        underperform = 1
    if row['Appear_rate'] <= threshold_App:
        underperform = 1
    return(underperform)

SAT_summary['Underperforming']  = SAT_summary.apply(lambda row:Assign_underperforming(row), axis= 1)
SAT_summary.sample(5).head()
class_size = pd.read_csv(class_size_path + '2010-2011-class-size-school-level-detail.csv')
class_size.head()
print("Class size with Nulls {}".format(class_size.shape))
class_size = class_size.loc[class_size['SCHOOLWIDE PUPIL-TEACHER RATIO'].notnull()]
print("Class size without Nulls {}".format(class_size.shape))
class_size = class_size.drop_duplicates(subset = ['SCHOOL NAME'])
class_size['SCHOOL NAME'] = list(map(to_upper, class_size['SCHOOL NAME']))
class_size['CSD'] = list(map(str, class_size['CSD']))
class_size.head(5)

def make_proper_school_code(row):
    """Function to a proper school code for class_size df"""
    code = ''
    if row['CSD'].__len__()==1:
        code = '0'+row['CSD']
    code = code + row['SCHOOL CODE']
    return(code)


class_size['school_code'] = class_size.apply(lambda row: make_proper_school_code(row), axis =1)


# Sanity check 
code1 = class_size.school_code.unique()
code2 = bayes_df['Location Code'].unique()
class_size.head()   

safety_df = pd.read_csv(safety_path+ '2010-2016-school-safety-report.csv')
safety_df.head()
aggregate_crime_stats = {'Major N':np.mean,
                         'Oth N':np.mean,
                         'NoCrim N':np.mean,
                         'Prop N':np.mean,
                         'Vio N':np.mean}
crime_summary = pd.DataFrame(safety_df.groupby(['Geographical District Code']).agg(aggregate_crime_stats))
crime_summary.reset_index(inplace = True)
crime_summary['Geographical District Code'] = list(map(int, crime_summary['Geographical District Code']))
print("The shape of locatio wise crime data is {}".format(crime_summary.shape))
crime_summary.head()
# Merge Other dataframes and make master_df
master_df = bayes_df.merge(SAT_summary, left_on = 'Location Code', right_on = 'DBN', how = 'left')
print(master_df.shape)
master_df.head()
# Merging class df for crime area wise summary 
master_df = master_df.merge(crime_summary, left_on = 'District', right_on = 'Geographical District Code', how = 'left')
print("Shape of master_df is {}".format(master_df.shape[0]))
master_df.head()

Cols_lp = [#'Community School?',
           'Economic Need Index', 
           'Percent ELL',
           #'Percent Asian',
           #'Percent Black',
           #'Percent Hispanic',
           #'Percent Black / Hispanic',
           #'Percent White',
           #'Student Attendance Rate',
           'Percent of Students Chronically Absent',
           #'Rigorous Instruction %',
           #'Rigorous Instruction Rating', 
           #'Student Achievement Rating',
           'Average ELA Proficiency',
           'Average Math Proficiency', 
           'location_cluster',
           'ELA_6_4s',
           'Math_6_4s',
           'ELA_7_4s',
           'Math_7_4s',
           'ELA_8_4s',
           'Math_8_4s', 
           'Underperforming', 
           'Major N']

can_be_used =['Collaborative Teachers %',
              'Collaborative Teachers Rating',
              'Supportive Environment %',
              'Supportive Environment Rating',
              'Effective School Leadership %',
              'Effective School Leadership Rating',
              'Strong Family-Community Ties %',
              'Strong Family-Community Ties Rating',
              'Trust %',
              'Trust Rating']

Cols_to_dummy =['Community School?',
                'Rigorous Instruction Rating'
               ]
def Make_dummies(dataframe, col_for_dummies):
    """ Function to make dummies for cat variables """
    col_for_svd = []
    df = dataframe.copy()
    for col in col_for_dummies:
        temp = pd.get_dummies(temp[col], prefix = col)
        col_for_svd = col_for_svd + temp.columns
        df.drop(col, inplace = True)
        df = df.concat([df, temp], axis = 0)
        print("{} and dummies added shape is {}".format(col, df.shape[1]))
    return(df, col_for_svd)

#LB_df, cols_for_svd = Make_dummies(bayes_df, col_for_dummies)
LB_df = master_df[Cols_lp].copy()
LB_df.head()
from sklearn.semi_supervised import LabelSpreading
def Label_propagator(df):
    """Function to use semi supervised label propagation to detect """
    df.fillna(-1, inplace = True)
    label_prop_model = LabelPropagation()
    label_prop_model = LabelSpreading(kernel = 'knn')
    features = [x for x in df.columns if x != 'Underperforming']
    label_prop_model.fit(df[features], df.Underperforming.values)
    predicted_label = label_prop_model.predict(df[features])
    df['Predicted_underperforming'] = predicted_label
    return(df)
pred_df = Label_propagator(LB_df)
pred_df.head()

pred_df['Predicted_underperforming'].describe()
f, axes = plt.subplots(2,figsize=(8, 8), sharex=True, sharey = True)
axes[0].hist(pred_df.Underperforming.values)
axes[1].hist(pred_df.Predicted_underperforming.values)
plt.show()
def plot_on_folium(df, status = 'before'):
    """function to generate map and add the pick up and drop coordinates
    1. Path = 1 : Join pickup (blue) and drop(red) using a straight line
    """
    circle=1
    df1 = df.copy()
    df1.reset_index(inplace = True)
    new_df = df1[['Latitude','Longitude','Predicted_underperforming', 'Underperforming', 'School Name']]
    #new_df.dropna(inplace = True)
    print(new_df.shape)
    m = folium.Map(location=[40.767937, -73.982155], zoom_start=10,tiles='Stamen Toner')
    for i in list(range(new_df.shape[0])):
        if status == 'before':
            #print(new_df.loc[i]['Underperforming'])
            if new_df.loc[i]['Underperforming']==-1.0:
                color_ = '#0000FF'
            elif new_df.loc[i]['Underperforming']==1.0:
                color_ = '#FF0000'
            else:
                color_ = '#00FF00'
        else:
            if new_df.loc[i]['Predicted_underperforming']==1.0:
                color_ = '#FF0000'
                #print(color_)
            else:
                color_ = '#00FF00'
        
        try:
            pick_long = new_df.loc[i]['Longitude']
            pick_lat = new_df.loc[i]['Latitude']
            pop = new_df.loc[i]['School Name']
            #print(pop)
            if circle == 1:
                folium.CircleMarker(location=[pick_lat, pick_long],
                        radius = 3,
                        color=color_).add_to(m)
            #folium.Marker([pick_lat, pick_long]).add_to(m)
        except:
            pass

    return(m)
map_schools_df = pd.concat([master_df[[x for x in master_df.columns if x != 'Underperforming']], pred_df[['Predicted_underperforming', 'Underperforming']]], axis =1)
print(map_schools_df.shape)

plot_on_folium(map_schools_df)
plot_on_folium(map_schools_df,status = 'after')
# Visualizing low and well performing schools on Principal componants will validate the Algorithm
from sklearn.decomposition import PCA
X = LB_df.values
pca = PCA(n_components=3)
PCA = pca.fit_transform(X)
print(pca.explained_variance_ratio_)
principalDf = pd.DataFrame(data = PCA
             , columns = ['PC1', 'PC2', 'PC3'])

finalDf = pd.concat([principalDf, map_schools_df[['Predicted_underperforming', 'School Name']]], axis = 1)

# 3D scatter plot for Total Fats
trace1 = go.Scatter3d(
    x=finalDf.PC1.values,
    y=finalDf.PC2.values,
    z=finalDf.PC3.values,
    text=finalDf['School Name'].values,
    mode='markers',
    marker=dict(
        sizemode='diameter',
#         sizeref=750,
#         size= dailyValue['Cholesterol (% Daily Value)'].values,
        color = finalDf['Predicted_underperforming'].values,
        colorscale = [[0.0, 'rgb(0,255,0)'], [1.0, 'rgb(255,0,0)']],
        colorbar = dict(title = 'Label propagation'),
        line=dict(color='rgb(255, 255, 255)')
    )
)

data=[trace1]
layout=dict(height=800, width=800, title='Underperforming schools visual')
fig=dict(data=data, layout=layout)
iplot(fig, filename='3DBubble')


# Saving the dataframes for further Analysis and as output
path_to_save_dfs = ''
bayes_df.to_csv(path_to_save_dfs + "Bayes_df.csv", index = False)
SAT_summary.to_csv(path_to_save_dfs + "SAT_summary.csv", index = False)
class_size.to_csv(path_to_save_dfs + "Class_size.csv", index = False)
safety_df.to_csv(path_to_save_dfs + "crime_df.csv", index = False)
map_schools_df.to_csv(path_to_save_dfs +"Labels.csv", index = False) # Contains School name and if the school is underperforming 
map_schools_df.to_csv(path_to_save_dfs +"Performance_of_schools.csv", index = False)
# final list of well-performing and underperforming schools
