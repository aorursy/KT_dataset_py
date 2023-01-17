import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

sns.set(color_codes=True)
%matplotlib inline
school_explorer=pd.read_csv("../input/data-science-for-good/2016 School Explorer.csv")
school_explorer.rename(columns=lambda x:x.replace('tested','Tested'),inplace=True)
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace(',', '')
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].str.replace('$', '')
school_explorer['School Income Estimate'] = school_explorer['School Income Estimate'].astype(float)

school_explorer["District"]=school_explorer["District"].astype(str)
school_explorer["Percent ELL"]=school_explorer["Percent ELL"].str.strip("%").astype(float)/100
school_explorer["Percent Asian"]=school_explorer["Percent Asian"].str.strip("%").astype(float)/100
school_explorer["Percent Black"]=school_explorer["Percent Black"].str.strip("%").astype(float)/100
school_explorer["Percent Hispanic"]=school_explorer["Percent Hispanic"].str.strip("%").astype(float)/100
school_explorer["Percent Black / Hispanic"]=school_explorer["Percent Black / Hispanic"].str.strip("%").astype(float)/100
school_explorer["Percent White"]=school_explorer["Percent White"].str.strip("%").astype(float)/100
school_explorer["Student Attendance Rate"]=school_explorer["Student Attendance Rate"].str.strip("%").astype(float)/100
school_explorer["Percent of Students Chronically Absent"]=school_explorer["Percent of Students Chronically Absent"].str.strip("%").astype(float)/100
school_explorer["Rigorous Instruction %"]=school_explorer["Rigorous Instruction %"].str.strip("%").astype(float)/100
school_explorer["Collaborative Teachers %"]=school_explorer["Collaborative Teachers %"].str.strip("%").astype(float)/100
school_explorer["Supportive Environment %"]=school_explorer["Supportive Environment %"].str.strip("%").astype(float)/100
school_explorer["Effective School Leadership %"]=school_explorer["Effective School Leadership %"].str.strip("%").astype(float)/100
school_explorer["Strong Family-Community Ties %"]=school_explorer["Strong Family-Community Ties %"].str.strip("%").astype(float)/100
school_explorer["Trust %"]=school_explorer["Trust %"].str.strip("%").astype(float)/100
def fun(df):
    if df["Grade 8 Math - All Students Tested"]!=0 and df["Grade 8 ELA - All Students Tested"]!=0:
        return "Group1"
    elif ("0K" not in df["Grade High"]) and int(df["Grade High"])>=8 and ((df["Grade 8 ELA - All Students Tested"]==0) or (df["Grade 8 Math - All Students Tested"]==0)):
        return "Group2"
    else:
        return "Other"
    
school_explorer["Class"]=school_explorer.apply(fun,axis=1)

print("Group 1 #: {} ({:.2f}%)".format(len(school_explorer[school_explorer["Class"]=="Group1"]),len(school_explorer[school_explorer["Class"]=="Group1"])/len(school_explorer)*100))
print("Group 2 #: {} ({:.2f}%)".format(len(school_explorer[school_explorer["Class"]=="Group2"]),len(school_explorer[school_explorer["Class"]=="Group2"])/len(school_explorer)*100))
print("Group Other #: {} ({:.2f}%)".format(len(school_explorer[school_explorer["Class"]=="Other"]),len(school_explorer[school_explorer["Class"]=="Other"])/len(school_explorer)*100))
school_explorer_sub=school_explorer[school_explorer["Class"]!="Other"]
plt.figure(figsize=(30,10))
ax=plt.subplot(121)
group1=school_explorer_sub[school_explorer_sub["Class"]=="Group1"].groupby(["City","Community School?"]).size()
group1=group1.unstack(fill_value=0)
group1["total"]=group1["No"]+group1["Yes"]
group1.sort_values(by="total",ascending=False,inplace=True)
group1.reset_index(inplace=True)

sns.set_color_codes("pastel")
sns.barplot(x=group1["total"][:10], y=group1["City"][:10],label="# Schools", color="b")
sns.set_color_codes("muted")
sns.barplot(x=group1["Yes"][:10], y=group1["City"][:10],label="# Community Schools", color="b")
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set_title("School Distribution by City (Top 10) for Group1",size=15)
ax.set_xlabel("count")

ax=plt.subplot(122)
group1=school_explorer_sub[school_explorer_sub["Class"]=="Group2"].groupby(["City","Community School?"]).size()
group1=group1.unstack(fill_value=0)
group1["total"]=group1["No"]+group1["Yes"]
group1.sort_values(by="total",ascending=False,inplace=True)
group1.reset_index(inplace=True)

sns.set_color_codes("pastel")
sns.barplot(x=group1["total"][:10], y=group1["City"][:10],label="# Schools", color="b")

sns.set_color_codes("muted")
sns.barplot(x=group1["Yes"][:10], y=group1["City"][:10],label="# Community Schools", color="b")
ax.legend(ncol=2, loc="lower right", frameon=True)
ax.set_title("School Distribution by City (Top 10) for Group2",size=15)
ax.set_xlabel("count")
plt.show()

plt.figure(figsize=(30,10))
ax=plt.subplot(131)
plt.scatter(x="Longitude",y="Latitude",data=school_explorer_sub[school_explorer_sub["Class"]=="Group1"],
#             s=df1["Economic Need Index"]*40,
            c="orange",
            marker='v',
            label="Group1")
plt.scatter(x="Longitude",y="Latitude",data=school_explorer_sub[school_explorer_sub["Class"]=="Group2"],
            c="red",
            marker='s',
            label="Group2")
ax.set_xlabel("Logitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographical distribution", size=15)
ax.legend()

ax=plt.subplot(132)
plt.scatter(x=school_explorer_sub["Longitude"],y=school_explorer_sub["Latitude"],
            s=school_explorer_sub["School Income Estimate"]/500,
            c=["orange" if x=="Group1" else "red" for x in school_explorer_sub["Class"]],
            marker='s')
ax.set_xlabel("Logitude")
ax.set_ylabel("Latitude")
ax.set_title("School Incom Geographical distribution", size=15)

ax=plt.subplot(133)
plt.scatter(x=school_explorer_sub["Longitude"],y=school_explorer_sub["Latitude"],
            s=school_explorer_sub["Economic Need Index"]*40,
            c=["orange" if x=="Group1" else "red" for x in school_explorer_sub["Class"]],
            marker='s')
ax.set_xlabel("Logitude")
ax.set_ylabel("Latitude")
ax.set_title("ENI Geographical distribution", size=15)

plt.figure(figsize=(30,10))
ax=plt.subplot(121)
sns.boxplot(y='School Income Estimate',x="Class", hue="Community School?",data=school_explorer_sub,palette="Set2")
ax.set_title("School Incom Estimate",size=15)

ax=plt.subplot(122)
sns.boxplot(y="Economic Need Index",x="Class", data=school_explorer_sub,palette="Set3")
plt.title("Economic Need Index distribution on the 2 Groups")
plt.show()

plt.figure(figsize=(20,8))
ax=plt.subplot(111)
sns.boxplot(y='School Income Estimate',x="District",data=school_explorer_sub)
ax.set_title('Income vs District', size=15)
plt.show()

plt.figure(figsize=(20,8))
ax=plt.subplot(111)
sns.boxplot(y='Economic Need Index',x="District",data=school_explorer_sub)
ax.set_title('ENI vs District', size=15)
plt.show()
tmp=school_explorer_sub[["Latitude","Longitude","Percent Asian","Percent Black","Percent Hispanic","Percent White","Class"]]

f,axes=plt.subplots(2,4,figsize=(18,8),sharex=True)

ax=axes[0,0]
sns.kdeplot(tmp[tmp["Class"]=="Group1"]["Percent Asian"] , color=sns.color_palette("Set1")[0],shade=True, ax=ax,label='Group1')
sns.kdeplot(tmp[tmp["Class"]=="Group2"]["Percent Asian"] , color=sns.color_palette("Set1")[1],shade=True, ax=ax,label='Group2')
ax.set_title("# Asian student distribution" ,size=13)

ax=axes[0,1]
sns.kdeplot(tmp[tmp["Class"]=="Group1"]["Percent Black"] , color=sns.color_palette("Set1")[0],shade=True, ax=ax,label='Group1')
sns.kdeplot(tmp[tmp["Class"]=="Group2"]["Percent Black"] , color=sns.color_palette("Set1")[1],shade=True, ax=ax,label='Group2')
ax.set_title("# Black student distribution",size=13)

ax=axes[0,2]
sns.kdeplot(tmp[tmp["Class"]=="Group1"]["Percent Hispanic"] , color=sns.color_palette("Set1")[0],shade=True, ax=ax,label='Group1')
sns.kdeplot(tmp[tmp["Class"]=="Group2"]["Percent Hispanic"] , color=sns.color_palette("Set1")[1],shade=True, ax=ax,label='Group2')
ax.set_title("# Hispanic student distribution",size=13)

ax=axes[0,3]
sns.kdeplot(tmp[tmp["Class"]=="Group1"]["Percent White"] , color=sns.color_palette("Set1")[0],shade=True, ax=ax,label='Group1')
sns.kdeplot(tmp[tmp["Class"]=="Group2"]["Percent White"] , color=sns.color_palette("Set1")[1],shade=True, ax=ax,label='Group2')
ax.set_title("# White student distribution",size=13)

ax=axes[1,0]
sns.boxplot(y="Percent Asian", x="Class", data=tmp , ax=ax)
ax=axes[1,1]
sns.boxplot(y="Percent Black", x="Class", data=tmp , ax=ax)
ax=axes[1,2]
sns.boxplot(y="Percent Hispanic", x="Class", data=tmp , ax=ax)
ax=axes[1,3]
sns.boxplot(y="Percent White", x="Class", data=tmp , ax=ax)
plt.show()
from scipy.stats import ttest_ind

print("T-test for Percent Asian of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Percent Asian"],
                tmp[tmp["Class"]=="Group2"]["Percent Asian"],
                equal_var = False))

print('-'*50)
print("T-test for Percent Black of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Percent Black"],
                tmp[tmp["Class"]=="Group2"]["Percent Black"],
                equal_var = False))
print('-'*50)
print("T-test for Percent Hispanic of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Percent Hispanic"],
                tmp[tmp["Class"]=="Group2"]["Percent Hispanic"],
                equal_var = False))

print('-'*50)
print("T-test for Percent White of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Percent White"],
                tmp[tmp["Class"]=="Group2"]["Percent White"],
                equal_var = False))

tmp=school_explorer_sub[["Rigorous Instruction %","Rigorous Instruction Rating","Collaborative Teachers %","Collaborative Teachers Rating",
                         "Supportive Environment %","Supportive Environment Rating","Effective School Leadership %","Effective School Leadership Rating",
                         "Strong Family-Community Ties %","Strong Family-Community Ties Rating","Trust %","Trust Rating","Class"]]
# tmp.head(3)
plt.figure(figsize=(15,10))
plt.subplots_adjust(wspace=0.5)
ax=plt.subplot(231)
sns.boxplot(y="Rigorous Instruction %",x="Class",data=tmp,color=sns.color_palette("Set2")[0])
ax.set_title("Rigorous Instruction % Distribution", size=12)
ax=plt.subplot(232)
sns.boxplot(y="Collaborative Teachers %",x="Class",data=tmp,color=sns.color_palette("Set2")[1])
ax.set_title("Collaborative Teachers % Distribution", size=12)
ax=plt.subplot(233)
sns.boxplot(y="Supportive Environment %",x="Class",data=tmp,color=sns.color_palette("Set2")[2])
ax.set_title("Supportive Environment % Distribution", size=12)
ax=plt.subplot(234)
sns.boxplot(y="Effective School Leadership %",x="Class",data=tmp,color=sns.color_palette("Set2")[3])
ax.set_title("Effective School Leadership % Distribution", size=12)
ax=plt.subplot(235)
sns.boxplot(y="Strong Family-Community Ties %",x="Class",data=tmp,color=sns.color_palette("Set2")[4])
ax.set_title("Strong Family-Community Ties % Distribution", size=12)
ax=plt.subplot(236)
sns.boxplot(y="Trust %",x="Class",data=tmp,color=sns.color_palette("Set2")[5])
ax.set_title("Trust % Distribution", size=12)
plt.show()

plt.figure(figsize=(15,10))
plt.subplots_adjust(wspace=0.7)
ax=plt.subplot(231)
tmp1=tmp.groupby(["Class","Rigorous Instruction Rating"]).size().unstack(0)
tmp1=tmp1/tmp1.sum(axis=0)
tmp1=tmp1.stack().reset_index().rename(columns={0:'percentage'})
sns.barplot(y="Rigorous Instruction Rating",x='percentage',hue="Class",data=tmp1,color=sns.color_palette("Set2")[0])
ax.set_title("Rigorous Instruction Rating Distribution", size=12)

ax=plt.subplot(232)
tmp1=tmp.groupby(["Class","Collaborative Teachers Rating"]).size().unstack(0)
tmp1=tmp1/tmp1.sum(axis=0)
tmp1=tmp1.stack().reset_index().rename(columns={0:'percentage'})
sns.barplot(y="Collaborative Teachers Rating",x='percentage',hue="Class",data=tmp1,color=sns.color_palette("Set2")[1])
ax.set_title("Collaborative Teachers Rating Distribution", size=12)

ax=plt.subplot(233)
tmp1=tmp.groupby(["Class","Supportive Environment Rating"]).size().unstack(0)
tmp1=tmp1/tmp1.sum(axis=0)
tmp1=tmp1.stack().reset_index().rename(columns={0:'percentage'})
sns.barplot(y="Supportive Environment Rating",x='percentage',hue="Class",data=tmp1,color=sns.color_palette("Set2")[2])
ax.set_title("Supportive Environment Rating Distribution", size=12)

ax=plt.subplot(234)
tmp1=tmp.groupby(["Class","Effective School Leadership Rating"]).size().unstack(0)
tmp1=tmp1/tmp1.sum(axis=0)
tmp1=tmp1.stack().reset_index().rename(columns={0:'percentage'})
sns.barplot(y="Effective School Leadership Rating",x='percentage',hue="Class",data=tmp1,color=sns.color_palette("Set2")[3])
ax.set_title("Effective School Rating Distribution", size=12)

ax=plt.subplot(235)
tmp1=tmp.groupby(["Class","Strong Family-Community Ties Rating"]).size().unstack(0)
tmp1=tmp1/tmp1.sum(axis=0)
tmp1=tmp1.stack().reset_index().rename(columns={0:'percentage'})
sns.barplot(y="Strong Family-Community Ties Rating",x='percentage',hue="Class",data=tmp1,color=sns.color_palette("Set2")[4])
ax.set_title("Strong Family-Community Ties Rating Distribution", size=12)

ax=plt.subplot(236)
tmp1=tmp.groupby(["Class","Trust Rating"]).size().unstack(0)
tmp1=tmp1/tmp1.sum(axis=0)
tmp1=tmp1.stack().reset_index().rename(columns={0:'percentage'})
sns.barplot(y="Trust Rating",x='percentage',hue="Class",data=tmp1,color=sns.color_palette("Set2")[5])
ax.set_title("Trust Rating Distribution", size=12)
plt.show()
tmp.fillna(0,inplace=True)
from scipy.stats import ttest_ind
print("T-test for Rigorous Instruction % of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Rigorous Instruction %"],
                tmp[tmp["Class"]=="Group2"]["Rigorous Instruction %"],
                equal_var = False))

print('-'*50)
print("T-test for Collaborative Teachers % of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Collaborative Teachers %"],
                tmp[tmp["Class"]=="Group2"]["Collaborative Teachers %"],
                equal_var = False))
print('-'*50)
print("T-test for Supportive Environment % of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Supportive Environment %"],
                tmp[tmp["Class"]=="Group2"]["Supportive Environment %"],
                equal_var = False))

print('-'*50)
print("T-test for Effective School Leadership % of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Effective School Leadership %"],
                tmp[tmp["Class"]=="Group2"]["Effective School Leadership %"],
                equal_var = False))

print('-'*50)
print("T-test for Strong Family-Community Ties % of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Strong Family-Community Ties %"],
                tmp[tmp["Class"]=="Group2"]["Strong Family-Community Ties %"],
                equal_var = False))

print('-'*50)
print("T-test for Trust % of Group1 and Group2:")
print(ttest_ind(tmp[tmp["Class"]=="Group1"]["Trust %"],
                tmp[tmp["Class"]=="Group2"]["Trust %"],
                equal_var = False))
plt.figure(figsize=(20,6))
ax=plt.subplot(131)
group2=school_explorer_sub.groupby(["Class",'Student Achievement Rating']).size().unstack(1).fillna(0)
group2=group2.apply(lambda x:x/group2.sum(axis=1)).stack().reset_index()
group2=group2.rename(columns={0:'pct'})
sns.barplot(x='Student Achievement Rating',y='pct',hue="Class", data=group2,order=["Exceeding Target","Approaching Target","Meeting Target","Not Meeting Target"], palette='Set1')
ax.set_title("Student Achievement Rating",size=15)

ax=plt.subplot(132)
sns.boxplot(y="Average ELA Proficiency",x="Class",data=school_explorer_sub,palette="Set2")
ax.set_title("Average ELA Proficiency",size=15)

ax=plt.subplot(133)
sns.boxplot(y="Average Math Proficiency",x="Class",data=school_explorer_sub,palette="Set3")
ax.set_title("Average Math Proficiency",size=15)

plt.show()
plt.figure(figsize=(15,10))
ax=plt.subplot(221)
sns.kdeplot(school_explorer_sub[school_explorer_sub['Class']=='Group1']['Student Attendance Rate'] , color='b',shade=True, label='SAR for Group1')
sns.kdeplot(school_explorer_sub[school_explorer_sub['Class']=='Group2']['Student Attendance Rate'] , color='y',shade=True, label='SAR for Group2')
ax.set_title('Student Attendance Rate Distributon',size=15)
ax.set_xlabel('Student Attendance Rate')
ax.set_ylabel('Frequency')

ax=plt.subplot(222)
sns.kdeplot(school_explorer_sub[school_explorer_sub['Class']=='Group1']['Percent of Students Chronically Absent'] , color='b',shade=True, label='Students Chronically Absent % for Group1')
sns.kdeplot(school_explorer_sub[school_explorer_sub['Class']=='Group2']['Percent of Students Chronically Absent'] , color='y',shade=True, label='Students Chronically Absent % for Group2')
ax.set_title('Percent of Students Chronically Absent Distributon',size=15)
ax.set_xlabel('Percent of Students Chronically Absent')
ax.set_ylabel('Frequency')

ax=plt.subplot(223)
sns.boxplot(y="Student Attendance Rate",x='Class',data=school_explorer_sub,palette="Set1")
ax.set_title("Student Attendace Rate for 2 Groups",size=15)

ax=plt.subplot(224)
sns.boxplot(y="Percent of Students Chronically Absent",x='Class',data=school_explorer_sub,palette="Set1")
ax.set_title("Percent of Students Chronically Absent for 2 Groups",size=15)
plt.show()
group_list=["Group1","Group2"]
indexes=["Student Attendance Rate","Percent of Students Chronically Absent"]

output=[]
for idx in indexes:
    for group in group_list:    
        doc_len_list=school_explorer_sub[school_explorer_sub["Class"]==group][idx].dropna()

        Q1=np.percentile(doc_len_list,25)
        Q3=np.percentile(doc_len_list,75)
        deltaQ=Q3-Q1
        
        output.append([group,
                    idx,
                    np.min(doc_len_list),
                    Q1-1.5*deltaQ,
                    Q1,
                    np.mean(doc_len_list),
                    np.percentile(doc_len_list,50),
                    Q3,
                    Q3+1.5*deltaQ,
                    np.max(doc_len_list)])

output=pd.DataFrame(output,columns=["group","index","min","lower farout","Q1","mean","Q2","Q3","upper farout","max"])


print("T-test for Student Attendance Rate of Group1 and Group2:")
print(ttest_ind(school_explorer_sub[school_explorer_sub["Class"]=="Group1"]["Student Attendance Rate"].dropna(),
                school_explorer_sub[school_explorer_sub["Class"]=="Group2"]["Student Attendance Rate"].dropna(),
                equal_var = False))

print('-'*50)
print("T-test for Percent of Students Chronically Absent of Group1 and Group2:")
print(ttest_ind(school_explorer_sub[school_explorer_sub["Class"]=="Group1"]["Percent of Students Chronically Absent"].dropna(),
                school_explorer_sub[school_explorer_sub["Class"]=="Group2"]["Percent of Students Chronically Absent"].dropna(),
                equal_var = False))

output
school_explorer_sub[(school_explorer_sub["Student Attendance Rate"]<0.85) & (school_explorer_sub["Percent of Students Chronically Absent"]>0.57)][["School Name","Student Attendance Rate","Percent of Students Chronically Absent","Community School?","Economic Need Index","Class"]]
Demographic_Snapshot_School=pd.read_csv("../input/2013-2018-demographic-snapshot-district/2013_-_2018_Demographic_Snapshot_School.csv")
Demographic_Snapshot_School_16=Demographic_Snapshot_School[Demographic_Snapshot_School["Year"]=="2015-16"]
# Demographic_Snapshot_School_16.head()
school_categorization=school_explorer_sub[["School Name","Location Code","District","City","Latitude","Longitude",
                                           "Grade 8 ELA - All Students Tested","Grade 8 ELA 4s - All Students",
                                           "Grade 8 ELA 4s - American Indian or Alaska Native",
                                           "Grade 8 ELA 4s - Black or African American","Grade 8 ELA 4s - Hispanic or Latino",
                                           "Grade 8 ELA 4s - Asian or Pacific Islander","Grade 8 ELA 4s - White",
                                           "Grade 8 ELA 4s - Multiracial","Grade 8 ELA 4s - Limited English Proficient",
                                           "Grade 8 ELA 4s - Economically Disadvantaged","Grade 8 Math - All Students Tested",
                                           "Grade 8 Math 4s - All Students","Grade 8 Math 4s - American Indian or Alaska Native",
                                           "Grade 8 Math 4s - Black or African American",
                                           "Grade 8 Math 4s - Hispanic or Latino","Grade 8 Math 4s - Asian or Pacific Islander",
                                           "Grade 8 Math 4s - White","Grade 8 Math 4s - Multiracial",
                                           "Grade 8 Math 4s - Limited English Proficient","Grade 8 Math 4s - Economically Disadvantaged"]]

school_categorization=pd.merge(school_categorization,Demographic_Snapshot_School_16,left_on="Location Code",right_on="DBN")
school_categorization.head(3)
# tmp.rename(columns=lambda x:x.replace("Grade 8 ",""),inplace=True)
school_categorization["Grade 8 ELA Participation Rate"]=school_categorization["Grade 8 ELA - All Students Tested"]/school_categorization["Grade 8"]
school_categorization["Grade 8 ELA Excellent Rate"]=school_categorization["Grade 8 ELA 4s - All Students"]/school_categorization["Grade 8 ELA - All Students Tested"]
school_categorization["Grade 8 ELA Excellent Rate for American Indian or Alaska Native"]=school_categorization["Grade 8 ELA 4s - American Indian or Alaska Native"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 ELA Excellent Rate for Black or African American"]=school_categorization["Grade 8 ELA 4s - Black or African American"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 ELA Excellent Rate for Hispanic or Latino"]=school_categorization["Grade 8 ELA 4s - Hispanic or Latino"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 ELA Excellent Rate for Asian or Pacific Islander"]=school_categorization["Grade 8 ELA 4s - Asian or Pacific Islander"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 ELA Excellent Rate for White"]=school_categorization["Grade 8 ELA 4s - White"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 ELA Excellent Rate for Multiracial"]=school_categorization["Grade 8 ELA 4s - Multiracial"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 ELA Excellent Rate for Limited English Proficient"]=school_categorization["Grade 8 ELA 4s - Limited English Proficient"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 ELA Excellent Rate for Economically Disadvantaged"]=school_categorization["Grade 8 ELA 4s - Economically Disadvantaged"]/school_categorization["Grade 8 ELA 4s - All Students"]

school_categorization["Grade 8 Math Participation Rate"]=school_categorization["Grade 8 Math - All Students Tested"]/school_categorization["Grade 8"]
school_categorization["Grade 8 Math Excellent Rate"]=school_categorization["Grade 8 Math 4s - All Students"]/school_categorization["Grade 8 Math - All Students Tested"]
school_categorization["Grade 8 Math Excellent Rate for American Indian or Alaska Native"]=school_categorization["Grade 8 Math 4s - American Indian or Alaska Native"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for Black or African American"]=school_categorization["Grade 8 Math 4s - Black or African American"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for Hispanic or Latino"]=school_categorization["Grade 8 Math 4s - Hispanic or Latino"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for Asian or Pacific Islander"]=school_categorization["Grade 8 Math 4s - Asian or Pacific Islander"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for White"]=school_categorization["Grade 8 Math 4s - White"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for Multiracial"]=school_categorization["Grade 8 Math 4s - Multiracial"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for Limited English Proficient"]=school_categorization["Grade 8 Math 4s - Limited English Proficient"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for Economically Disadvantaged"]=school_categorization["Grade 8 Math 4s - Economically Disadvantaged"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization.fillna(0,inplace=True)

school_categorization.sample(5)
print("# Not equal records for ELA: {}".format(len(school_categorization[school_categorization["Grade 8 ELA 4s - All Students"]>school_categorization["Grade 8 ELA 4s - American Indian or Alaska Native"]+school_categorization["Grade 8 ELA 4s - Black or African American"]+school_categorization["Grade 8 ELA 4s - Hispanic or Latino"]+school_categorization["Grade 8 ELA 4s - Asian or Pacific Islander"]+school_categorization["Grade 8 ELA 4s - White"]+school_categorization["Grade 8 ELA 4s - Multiracial"]])))
print("# Not equal records for Math: {}".format(len(school_categorization[school_categorization["Grade 8 Math 4s - All Students"]>school_categorization["Grade 8 Math 4s - American Indian or Alaska Native"]+school_categorization["Grade 8 Math 4s - Black or African American"]+school_categorization["Grade 8 Math 4s - Hispanic or Latino"]+school_categorization["Grade 8 Math 4s - Asian or Pacific Islander"]+school_categorization["Grade 8 Math 4s - White"]+school_categorization["Grade 8 Math 4s - Multiracial"]])))
school_categorization["Grade 8 ELA 4s - Unknown"]=school_categorization["Grade 8 ELA 4s - All Students"]-(school_categorization["Grade 8 ELA 4s - American Indian or Alaska Native"]+school_categorization["Grade 8 ELA 4s - Black or African American"]+school_categorization["Grade 8 ELA 4s - Hispanic or Latino"]+school_categorization["Grade 8 ELA 4s - Asian or Pacific Islander"]+school_categorization["Grade 8 ELA 4s - White"]+school_categorization["Grade 8 ELA 4s - Multiracial"])
school_categorization["Grade 8 Math 4s - Unknown"]=school_categorization["Grade 8 Math 4s - All Students"]-(school_categorization["Grade 8 Math 4s - American Indian or Alaska Native"]+school_categorization["Grade 8 Math 4s - Black or African American"]+school_categorization["Grade 8 Math 4s - Hispanic or Latino"]+school_categorization["Grade 8 Math 4s - Asian or Pacific Islander"]+school_categorization["Grade 8 Math 4s - White"]+school_categorization["Grade 8 Math 4s - Multiracial"])

school_categorization["Grade 8 ELA Excellent Rate for Unknown"]=school_categorization["Grade 8 ELA 4s - Unknown"]/school_categorization["Grade 8 ELA 4s - All Students"]
school_categorization["Grade 8 Math Excellent Rate for Unknown"]=school_categorization["Grade 8 Math 4s - Unknown"]/school_categorization["Grade 8 Math 4s - All Students"]
school_categorization.fillna(0,inplace=True)
plt.figure(figsize=(16,4))
ax=plt.subplot(141)
sns.boxplot(y='Grade 8 ELA Participation Rate',data=school_categorization)
ax.set_title('ELA Participation Rate', size=15)

ax=plt.subplot(142)
sns.boxplot(y='Grade 8 ELA Excellent Rate',data=school_categorization)
ax.set_title('ELA Excellent Rate', size=15)

ax=plt.subplot(143)
sns.boxplot(y='Grade 8 Math Participation Rate',data=school_categorization)
ax.set_title('Math Participation Rate', size=15)

ax=plt.subplot(144)
sns.boxplot(y='Grade 8 Math Excellent Rate',data=school_categorization)
ax.set_title('Math Excellent Rate', size=15)
plt.show()
# caculate threshold
quantile_ela_part=school_categorization["Grade 8 ELA Participation Rate"].mean()
quantile_ela_excellent=school_categorization["Grade 8 ELA Excellent Rate"].mean()
quantile_math_part=school_categorization["Grade 8 Math Participation Rate"].mean()
quantile_math_excellent=school_categorization["Grade 8 Math Excellent Rate"].mean()


# plot
plt.figure(figsize=(12,6))
ax=plt.subplot(121)
plt.scatter(x=school_categorization["Grade 8 ELA Participation Rate"],y=school_categorization["Grade 8 ELA Excellent Rate"],color="red")
plt.xlabel("ELA Participation Rate")
plt.ylabel("ELA Excellent Rate")
plt.hlines(quantile_ela_excellent,0,1,color='k',linestyles='dotted')
plt.vlines(quantile_ela_part,0,1,color='k',linestyles='dotted')
plt.text(0.85,0.75,"Star",color="blue")
plt.text(0.85,0.05,"Question mark",color="blue")
plt.text(0.05,0.75,"Cash cow",color="blue")
plt.text(0.05,0.05,"Dog",color="blue")
plt.xlim(0,1)
plt.ylim(0,1)
plt.title("Participation rate vs Excellent rate for ELA",size=15)

ax=plt.subplot(122)
plt.scatter(x=school_categorization["Grade 8 Math Participation Rate"],y=school_categorization["Grade 8 Math Excellent Rate"],color="orange")
plt.xlabel("Math participate pct")
plt.ylabel("Math 4s pct")
plt.hlines(quantile_math_excellent,0,1,color='k',linestyles='dotted')
plt.vlines(quantile_math_part,0,1,color='k',linestyles='dotted')
plt.text(0.75,0.75,"Star",color="blue")
plt.text(0.75,0.05,"Question mark",color="blue")
plt.text(0.05,0.75,"Cash cow",color="blue")
plt.text(0.05,0.05,"Dog",color="blue")
plt.xlim(0,1)
plt.ylim(0,1)
plt.title("Participation rate vs Excellent rate for Math",size=15)
plt.show()
def fun_ela(tmp):
    if tmp["Grade 8 ELA Participation Rate"]>=quantile_ela_part and tmp["Grade 8 ELA Excellent Rate"]>=quantile_ela_excellent:
        return "Star"
    elif tmp["Grade 8 ELA Participation Rate"]<=quantile_ela_part and tmp["Grade 8 ELA Excellent Rate"]>=quantile_ela_excellent:
        return "Cash cow"
    elif tmp["Grade 8 ELA Participation Rate"]>=quantile_ela_part and tmp["Grade 8 ELA Excellent Rate"]<=quantile_ela_excellent:
        return "Question mark"
    elif tmp["Grade 8 ELA Participation Rate"]<=quantile_ela_part and tmp["Grade 8 ELA Excellent Rate"]<=quantile_ela_excellent:
        return "Dog"
    
school_categorization["ELA tag"]=school_categorization.apply(fun_ela,axis=1)

def fun_math(tmp):
    if tmp["Grade 8 Math Participation Rate"]>=quantile_math_part and tmp["Grade 8 Math Excellent Rate"]>=quantile_math_excellent:
        return "Star"
    elif tmp["Grade 8 Math Participation Rate"]<=quantile_math_part and tmp["Grade 8 Math Excellent Rate"]>=quantile_math_excellent:
        return "Cash cow"
    elif tmp["Grade 8 Math Participation Rate"]>=quantile_math_part and tmp["Grade 8 Math Excellent Rate"]<=quantile_math_excellent:
        return "Question mark"
    elif tmp["Grade 8 Math Participation Rate"]<=quantile_math_part and tmp["Grade 8 Math Excellent Rate"]<=quantile_math_excellent:
        return "Dog"
    
school_categorization["Math tag"]=school_categorization.apply(fun_math,axis=1)
print("Star schools:")
print(school_categorization[(school_categorization["Grade 8 ELA Participation Rate"]>=quantile_ela_part) & (school_categorization["Grade 8 ELA Excellent Rate"]>=quantile_ela_excellent)]["School Name_x"].head(3))
print("-"*70)

print("Cash cow schools:")
print(school_categorization[(school_categorization["Grade 8 ELA Participation Rate"]<=quantile_ela_part) & (school_categorization["Grade 8 ELA Excellent Rate"]>=quantile_ela_excellent)]["School Name_x"].head(3))
print("-"*70)

print("Question mark schools:")
print(school_categorization[(school_categorization["Grade 8 ELA Participation Rate"]>=quantile_ela_part) & (school_categorization["Grade 8 ELA Excellent Rate"]<=quantile_ela_excellent)]["School Name_x"].head(3))
print("-"*70)

print("Dog schools:")
print(school_categorization[(school_categorization["Grade 8 ELA Participation Rate"]<=quantile_ela_part) & (school_categorization["Grade 8 ELA Excellent Rate"]<=quantile_ela_excellent)]["School Name_x"].head(3))
print("-"*70)
plt.figure(figsize=(12,10))
plt.subplot(221)
sns.countplot(x="ELA tag",data=school_categorization,palette="Set3",order=["Star","Cash cow","Dog","Question mark"])
plt.title("Number of different type of schools for ELA", size=15)
plt.subplot(222)
sns.countplot(x="Math tag",data=school_categorization,palette="Set3",order=["Star","Cash cow","Dog","Question mark"])
plt.title("Number of different type of schools for Math", size=15)

ax=plt.subplot(223)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["ELA tag"]=="Star"],c="red",marker='v',label="Star",alpha=1)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["ELA tag"]=="Cash cow"],c="blue",marker='o',label="Cash cow",alpha=0.8)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["ELA tag"]=="Dog"],c="orange",marker='1',label="Dog",alpha=0.6)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["ELA tag"]=="Question mark"],c="green",marker='s',label="Question mark",alpha=0.4)
ax.set_xlabel("Logitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographical distribution | ELA", size=15)
ax.legend()

ax=plt.subplot(224)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["Math tag"]=="Star"],c="red",marker='v',label="Star",alpha=1)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["Math tag"]=="Cash cow"],c="blue",marker='o',label="Cash cow",alpha=0.8)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["Math tag"]=="Dog"],c="orange",marker='1',label="Dog",alpha=0.6)
plt.scatter(x="Longitude",y="Latitude",data=school_categorization[school_categorization["Math tag"]=="Question mark"],c="green",marker='s',label="Question mark",alpha=0.4)
ax.set_xlabel("Logitude")
ax.set_ylabel("Latitude")
ax.set_title("Geographical distribution | Math", size=15)
ax.legend()
plt.show()
print("Both \"Star\" for ELA/Math #: {}".format(len(school_categorization[(school_categorization["ELA tag"]=="Star") & (school_categorization["Math tag"]=="Star")])))
print("Both \"Question mark\" for ELA/Math #: {}".format(len(school_categorization[(school_categorization["ELA tag"]=="Question mark") & (school_categorization["Math tag"]=="Question mark")])))
group=school_categorization.groupby(["City","ELA tag"]).size()
group=group.unstack().fillna(0)

print(group[group["Question mark"]>group["Star"]])

group=group.apply(lambda x:x/group.sum(axis=1))
group.reset_index(inplace=True)
group.sort_values(by="Star",ascending=False,inplace=True)

label=group["City"]
idx=np.arange(len(group))

plt.figure(figsize=(12,10))
plt.barh(idx,group["Star"],height=0.9,color="r",label="Star") 
plt.barh(idx,-group["Question mark"],height=0.9,label="Question mark") 
plt.yticks(idx,label)
plt.vlines(1,-1,40,color='k',linestyles='dotted')
plt.vlines(0.5,-1,40,color='k',linestyles='dotted')
plt.vlines(-0.5,-1,40,color='k',linestyles='dotted')
plt.vlines(-1,-1,40,color='k',linestyles='dotted')
plt.text(0.5,-2,"0.5",color="k")
plt.text(1,-2,"1",color="k")
plt.text(-0.5,-2,"-0.5",color="k")
plt.text(-1,-2,"-1",color="k")
plt.gca().invert_yaxis()
plt.ylabel("City")
plt.xlabel("Percentage of Star/Question schools in that City")
plt.legend()
plt.title("Star/Question mark school distribution by City", size=15)
plt.show() 
diversity=pd.merge(school_explorer,
                   Demographic_Snapshot_School_16,
                   left_on="Location Code",right_on="DBN")
diversity["% Black"]=diversity["% Black"].str.strip("%").astype(float)/100
diversity["% Hispanic"]=diversity["% Hispanic"].str.strip("%").astype(float)/100

diversity["% Target"]=diversity["% Black"]+diversity["% Hispanic"]
diversity["# Grade 3-8 total"]=diversity["Grade 3"]+diversity["Grade 4"]+diversity["Grade 5"]+\
                               diversity["Grade 6"]+diversity["Grade 7"]+diversity["Grade 8"]
diversity["# Grade 3-8 target"]=diversity["# Grade 3-8 total"]*diversity["% Target"]

diversity["# ELA Participation"]=diversity["Grade 3 ELA - All Students Tested"]+diversity["Grade 4 ELA - All Students Tested"]+diversity["Grade 5 ELA - All Students Tested"]+diversity["Grade 6 ELA - All Students Tested"]+diversity["Grade 7 ELA - All Students Tested"]+diversity["Grade 8 ELA - All Students Tested"]
diversity["# Math Participation"]=diversity["Grade 3 Math - All Students Tested"]+diversity["Grade 4 Math - All Students Tested"]+diversity["Grade 5 Math - All Students Tested"]+diversity["Grade 6 Math - All Students Tested"]+diversity["Grade 7 Math - All Students Tested"]+diversity["Grade 8 Math - All Students Tested"]
diversity["# ELA Participation target"]=diversity["# ELA Participation"]*diversity["# Grade 3-8 target"]/diversity["# Grade 3-8 total"]
diversity["# Math Participation target"]=diversity["# Math Participation"]*diversity["# Grade 3-8 target"]/diversity["# Grade 3-8 total"]
diversity["% ELA Participation target"]=diversity["# ELA Participation target"]/diversity["# ELA Participation"]
diversity["% Math Participation target"]=diversity["# Math Participation target"]/diversity["# Math Participation"]

diversity["# ELA 4s target"]=diversity["Grade 3 ELA 4s - Black or African American"]+diversity["Grade 3 ELA 4s - Hispanic or Latino"]+\
                              diversity["Grade 4 ELA 4s - Black or African American"]+diversity["Grade 4 ELA 4s - Hispanic or Latino"]+\
                              diversity["Grade 5 ELA 4s - Black or African American"]+diversity["Grade 5 ELA 4s - Hispanic or Latino"]+\
                              diversity["Grade 6 ELA 4s - Black or African American"]+diversity["Grade 6 ELA 4s - Hispanic or Latino"]+\
                              diversity["Grade 7 ELA 4s - Black or African American"]+diversity["Grade 7 ELA 4s - Hispanic or Latino"]+\
                              diversity["Grade 8 ELA 4s - Black or African American"]+diversity["Grade 8 ELA 4s - Hispanic or Latino"]

diversity["# Math 4s target"]=diversity["Grade 3 Math 4s - Black or African American"]+diversity["Grade 3 Math 4s - Hispanic or Latino"]+\
                              diversity["Grade 4 Math 4s - Black or African American"]+diversity["Grade 4 Math 4s - Hispanic or Latino"]+\
                              diversity["Grade 5 Math 4s - Black or African American"]+diversity["Grade 5 Math 4s - Hispanic or Latino"]+\
                              diversity["Grade 6 Math 4s - Black or African American"]+diversity["Grade 6 Math 4s - Hispanic or Latino"]+\
                              diversity["Grade 7 Math 4s - Black or African American"]+diversity["Grade 7 Math 4s - Hispanic or Latino"]+\
                              diversity["Grade 8 Math 4s - Black or African American"]+diversity["Grade 8 Math 4s - Hispanic or Latino"]
                    
diversity["% ELA 4s target"]=diversity["# ELA 4s target"]/(diversity["Grade 3 ELA 4s - All Students"]+diversity["Grade 4 ELA 4s - All Students"]+\
                                                          diversity["Grade 5 ELA 4s - All Students"]+diversity["Grade 6 ELA 4s - All Students"]+\
                                                          diversity["Grade 7 ELA 4s - All Students"]+diversity["Grade 8 ELA 4s - All Students"])                

diversity["% Math 4s target"]=diversity["# Math 4s target"]/(diversity["Grade 3 Math 4s - All Students"]+diversity["Grade 4 Math 4s - All Students"]+\
                                                          diversity["Grade 5 Math 4s - All Students"]+diversity["Grade 6 Math 4s - All Students"]+\
                                                          diversity["Grade 7 Math 4s - All Students"]+diversity["Grade 8 Math 4s - All Students"]) 

diversity.iloc[:,-13:].tail(10)
plt.figure(figsize=(20,6))
ax=plt.subplot(131)
sns.kdeplot(diversity['% ELA 4s target'], color='r',shade=True, label='% ELA 4s target')
sns.kdeplot(diversity['% Math 4s target'], color='b',shade=True, label='% Math 4s target')
ax.set_title('Student Attendance Rate Distributon by Race',size=15)
ax.set_xlabel('Student Attendance Rate')
ax.set_ylabel('Frequency')
ax=plt.subplot(132)
plt.scatter(diversity["% ELA Participation target"], diversity['% ELA 4s target'])
ax.set_xlabel("% ELA Participation for Black/Hispanic  (% Target)")
ax.set_ylabel("% ELA Excellent for Black/Hispanic")
ax.set_title("Participation vs Excellent for Black/Hispanic | ELA",size=15)

ax=plt.subplot(133)
plt.scatter(diversity["% Math Participation target"], diversity['% Math 4s target'])
ax.set_xlabel("% Math Participation for Black/Hispanic (% Target)")
ax.set_ylabel("% Math Excellent for Black/Hispanic")
ax.set_title("Participation vs Excellent for Black/Hispanic | Math",size=15)
plt.show()
Demographic_Snapshot_School=pd.read_csv("../input/2013-2018-demographic-snapshot-district/2013_-_2018_Demographic_Snapshot_School.csv")

Demographic_Snapshot_School["Year"]=Demographic_Snapshot_School["Year"].apply(lambda x:x.split("-")[0])
Demographic_Snapshot_School["District"]=Demographic_Snapshot_School["DBN"].apply(lambda x:x[:2])
Demographic_Snapshot_School["% Black"]=Demographic_Snapshot_School["% Black"].str.strip("%").astype(float)/100
Demographic_Snapshot_School["% Hispanic"]=Demographic_Snapshot_School["% Hispanic"].str.strip("%").astype(float)/100
Demographic_Snapshot_School["% Students with Disabilities"]=Demographic_Snapshot_School["% Students with Disabilities"].str.strip("%").astype(float)/100
Demographic_Snapshot_School["% Poverty"]=Demographic_Snapshot_School["% Poverty"].str.strip("%").astype(float)/100

plt.figure(figsize=(18,4))
ax=plt.subplot(131)
group1=Demographic_Snapshot_School.groupby(["Year"]).agg({"% Black":"mean",
                                                          "% Hispanic":"mean",
                                                          "% Students with Disabilities":"mean",
                                                          "% Poverty":"mean"}).reset_index()
plt.plot(group1["Year"],group1["% Black"],"black",label="% Black")
plt.plot(group1["Year"],group1["% Hispanic"],"blue",label="% Hispanic")
plt.plot(group1["Year"],group1["% Students with Disabilities"],"green",label="% Students with Disabilities")
plt.plot(group1["Year"],group1["% Poverty"],"orange",label="% Poverty")
plt.legend()
plt.xlabel("Year")
plt.ylabel("Percentage")
plt.title("Proportion vs Year",size=15)
plt.show()

group1=Demographic_Snapshot_School.groupby(["Year","District"]).agg({"% Black":"mean",
                                                          "% Hispanic":"mean",
                                                          "% Students with Disabilities":"mean",
                                                          "% Poverty":"mean"}).reset_index()
districts=group1["District"].unique()

f,axes=plt.subplots(6,6,figsize=(20,20),sharex=True, sharey=True)

for i in range(6):
    for j in range(6):
        sns.pointplot(group1[group1["District"]==districts[i*6+j]]["Year"],
                      group1[group1["District"]==districts[i*6+j]]["% Black"],
                      color="black",
                      ax=axes[i,j])
        sns.pointplot(group1[group1["District"]==districts[i*6+j]]["Year"],
                      group1[group1["District"]==districts[i*6+j]]["% Hispanic"],
                      color="blue",
                      ax=axes[i,j])
        axes[i,j].set_title("District: {}".format(districts[i*6+j]))
        if i*6+j+1==len(districts):
            break
            

plt.show()
school_safety=pd.read_csv("../input/ny-2010-2016-school-safety-report/2010-2016-school-safety-report.csv")
school_safety=school_safety[school_safety["School Year"]=="2015-16"]
school_safety["# Crime"]=school_safety["Major N"]+school_safety["Oth N"]+school_safety["NoCrim N"]+school_safety["Prop N"]+school_safety["Vio N"]

school_safety_merged=pd.merge(school_safety[["DBN","# Crime"]],diversity,left_on="DBN",right_on="Location Code")
school_safety_merged=school_safety_merged[["New?","DBN_x","# Crime","District","Latitude","Longitude","City",
                                           "Community School?","Economic Need Index_x","Student Attendance Rate",
                                           "Percent of Students Chronically Absent","Rigorous Instruction %","Rigorous Instruction Rating",
                                           "Collaborative Teachers %","Collaborative Teachers Rating","Supportive Environment %",
                                           "Supportive Environment Rating","Effective School Leadership %","Effective School Leadership Rating",
                                           "Strong Family-Community Ties %","Strong Family-Community Ties Rating","Trust %",
                                           "Trust Rating","Student Achievement Rating","Average ELA Proficiency","Average Math Proficiency",
                                           "% Poverty"]]

school_safety_merged=pd.merge(school_safety_merged,school_categorization[["Location Code", "ELA tag", "Math tag"]],left_on="DBN_x",right_on="Location Code")
school_safety_merged=school_safety_merged[school_safety_merged["# Crime"]>0]
school_safety_merged["% Poverty"]=school_safety_merged["% Poverty"].str.strip("%").astype(float)/100
school_safety_merged["New?"]=school_safety_merged["New?"].apply(lambda x:"Yes" if type(x)!=float else "No")
school_safety_merged["District"]=school_safety_merged["District"].astype(str)

print("length of school_safety_merged: {}".format(len(school_safety_merged)))
plt.figure(figsize=(15,4))
ax=plt.subplot(131)
plt.hist(school_safety_merged["# Crime"],bins=30)
ax.set_xlabel("# Crime")
ax.set_ylabel("# schools")
ax.set_title("# Crime distribution", size=15)

ax=plt.subplot(132)
tmp=school_safety_merged[["ELA tag","# Crime"]]
tmp=tmp.groupby("ELA tag").agg({"# Crime":["sum","count"]})
tmp["# avg crime per school"]=tmp.iloc[:,0]/tmp.iloc[:,1]
tmp.reset_index(inplace=True)
sns.barplot(x="ELA tag", y="# avg crime per school", data=tmp,palette="Set1", order=["Star","Cash cow","Dog","Question mark"])
ax.set_title("Avg crime count vs. School category | ELA", size=15)

ax=plt.subplot(133)
tmp=school_safety_merged[["Math tag","# Crime"]]
tmp=tmp.groupby("Math tag").agg({"# Crime":["sum","count"]})
tmp["# avg crime per school"]=tmp.iloc[:,0]/tmp.iloc[:,1]
tmp.reset_index(inplace=True)
sns.barplot(x="Math tag", y="# avg crime per school", data=tmp, palette="Set3", order=["Star","Cash cow","Dog","Question mark"])
ax.set_title("Avg crime count vs. School category | Math", size=15)

plt.show()

plt.figure(figsize=(12,12))
tmp=school_safety_merged[["# Crime","Economic Need Index_x","Student Attendance Rate","Percent of Students Chronically Absent",
                          'Rigorous Instruction %', 'Collaborative Teachers %', 'Supportive Environment %',
                          'Effective School Leadership %','Strong Family-Community Ties %','Trust %', 
                          'Student Achievement Rating', 'Average ELA Proficiency','Average Math Proficiency', '% Poverty']]

corrmat = tmp.corr()
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True,cmap=plt.cm.get_cmap('RdYlBu'))
plt.title("Correation Matrix",size=15)
plt.show()

plt.figure(figsize=(20,6))
ax=plt.subplot(111)
tmp=school_safety_merged[["# Crime","City"]]
tmp=tmp.groupby("City").agg({"# Crime":"count"}).reset_index()
sns.barplot(x="City",y="# Crime", data=tmp)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
ax.set_title("# Crime GEO Distribution", size=15)
plt.show()
import folium
from folium import plugins
from io import StringIO
import folium 

lat_0=40.7127
lon_0=-74.0059

maps = folium.Map([lat_0,lon_0], zoom_start=10.5,tiles='stamentoner') # OpenStreetMap, Mapbox Bright, Mapbox Control Room, Stamen, CartoDB
for lat, long, cnt in zip(school_safety_merged['Latitude'], school_safety_merged['Longitude'], school_safety_merged["# Crime"]):
    folium.Circle([lat, long],radius=cnt*30, color="gold", fill=True, fill_opacity=0.8).add_to(maps)
maps
high_school_directory=pd.read_csv("../input/nyc-high-school-directory/2016-doe-high-school-directory.csv",
                                  usecols=["dbn","shared_space","grade_span_min","grade_span_max","bus","subway",
                                           "total_students","school_type","language_classes",
                                           "advancedplacement_courses","diplomaendorsements","extracurricular_activities","psal_sports_boys",
                                           "psal_sports_girls","psal_sports_coed","school_sports","partner_cbo","partner_hospital",
                                           "partner_highered","partner_cultural","partner_nonprofit","partner_corporate","partner_financial",
                                           "partner_other","se_services","ell_programs","school_accessibility_description",
                                           "number_programs",
                                           "priority01","priority02","priority03","priority04","priority05","priority06","priority07","priority08","priority09","priority10",
                                           "Community Board","Council District","Census Tract"])

high_school_directory_merge=pd.merge(diversity,high_school_directory,left_on="Location Code",right_on="dbn")
print("high_school_directory_merge remain {} records of school_categorization ({:.2f}%).".format(len(high_school_directory_merge),len(high_school_directory_merge)/len(diversity)*100))
high_school_directory_merge["number of language_classes"]=high_school_directory_merge["language_classes"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
unique_language_classes=list(set([y.strip()  for x in high_school_directory_merge["language_classes"] if type(x)==str for y in x.split(',')]))
print("unique language classes: {}\n".format(unique_language_classes))

plt.figure(figsize=(16,14))
plt.subplots_adjust(hspace=0.5)

ax=plt.subplot(221)
# sns.barplot(x='cnt', y="number of language_classes",data=group, label="# Schools", color="b")
sns.countplot(x='number of language_classes',data=high_school_directory_merge,palette="Set1")
plt.title("Distribution: How mang language classes",size=15)

ax=plt.subplot(222)
group={k:0 for k in unique_language_classes}
for i in range(len(high_school_directory_merge)):
    if type(high_school_directory_merge.loc[i,"language_classes"])==str:
        for x in high_school_directory_merge.loc[i,"language_classes"].split(','):
            group[x.strip()]+=1
group=pd.DataFrame([group])
group=group.stack().reset_index().drop('level_0',axis=1).rename(columns={'level_1':'classes',0:'count'})
sns.barplot("classes","count",data=group)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
plt.title("Distribution: Which language classes",size=15)

ax=plt.subplot(223)
unique_advancedplacement_courses=list(set([y.strip()  for x in high_school_directory_merge["advancedplacement_courses"] if type(x)==str for y in x.split(',')]))
print("unique advanced placement courses: {}\n".format(unique_advancedplacement_courses))
high_school_directory_merge["advancedplacement_courses_contain_English"]=high_school_directory_merge["advancedplacement_courses"].apply(lambda x:"Yes" if type(x)==str and "English" in x else "No")
high_school_directory_merge["advancedplacement_courses_contain_Calculus"]=high_school_directory_merge["advancedplacement_courses"].apply(lambda x:"Yes" if type(x)==str and "Calculus" in x else "No")
high_school_directory_merge["advancedplacement_courses_contain_Spanish"]=high_school_directory_merge["advancedplacement_courses"].apply(lambda x:"Yes" if type(x)==str and "Spanish" in x else "No")

tmp=high_school_directory_merge[["advancedplacement_courses_contain_English","advancedplacement_courses_contain_Calculus","advancedplacement_courses_contain_Spanish"]]
tmp=tmp.stack().reset_index().drop('level_0',axis=1).rename(columns={'level_1':'Subject',0:'Contain?'})
tmp["Subject"]=tmp["Subject"].apply(lambda x:x.split('_')[3:][0])
sns.countplot(x="Subject",hue="Contain?",data=tmp,palette="Set3")
plt.title("advancedplacement_courses",size=15)


ax=plt.subplot(224)
unique_diplomaendorsements=list(set([y.strip()  for x in high_school_directory_merge["diplomaendorsements"] if type(x)==str for y in x.split(',')]))
print("unique diploma endorsements: {}\n".format(unique_diplomaendorsements))
group={k:0 for k in unique_diplomaendorsements}
for i in range(len(high_school_directory_merge)):
    if type(high_school_directory_merge.loc[i,"diplomaendorsements"])==str:
        for x in high_school_directory_merge.loc[i,"diplomaendorsements"].split(','):
            group[x.strip()]+=1
group=pd.DataFrame([group])
group=group.stack().reset_index().drop('level_0',axis=1).rename(columns={'level_1':'diploma',0:'count'})
sns.barplot("diploma","count",data=group)
plt.title("diploma endorsements",size=15)
plt.show()
from scipy.stats import ttest_ind
tmp=high_school_directory_merge[["% ELA 4s target","advancedplacement_courses_contain_English","advancedplacement_courses_contain_Spanish"]].fillna(0)

print("T-test for advanced placement courses--English:")
print(ttest_ind(tmp[tmp["advancedplacement_courses_contain_English"]=="Yes"]["% ELA 4s target"],
                tmp[tmp["advancedplacement_courses_contain_English"]=="No"]["% ELA 4s target"],
                equal_var = False))

print('-'*50)
print("T-test for advanced placement courses--Spanish:")
print(ttest_ind(tmp[tmp["advancedplacement_courses_contain_Spanish"]=="Yes"]["% ELA 4s target"],
                tmp[tmp["advancedplacement_courses_contain_Spanish"]=="No"]["% ELA 4s target"],
                equal_var = False))
print()
print("Avg Excellent Rate for Spanish==Yes: {}".format(np.mean(tmp[tmp["advancedplacement_courses_contain_Spanish"]=="Yes"]["% ELA 4s target"])))
print("Avg Excellent Rate for Spanish==No: {}".format(np.mean(tmp[tmp["advancedplacement_courses_contain_Spanish"]=="No"]["% ELA 4s target"])))
tmp=high_school_directory_merge[["% ELA 4s target","number of language_classes"]].fillna(0)
corrmat = tmp.corr()
plt.figure(figsize=(12,5))
ax=plt.subplot(121)
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True,cmap=plt.cm.get_cmap('RdYlBu'))
plt.title("Correation Matrix",size=15)

ax=plt.subplot(122)
sns.boxplot(x="number of language_classes",y="% ELA 4s target",data=tmp,palette="Set1")
plt.title("number of language_classes distribution",size=15)
plt.show()
tmp=high_school_directory_merge[["% Math 4s target","advancedplacement_courses_contain_Calculus"]].fillna(0)

print("T-test for advanced placement courses--Math:")
print(ttest_ind(tmp[tmp["advancedplacement_courses_contain_Calculus"]=="Yes"]["% Math 4s target"],
                tmp[tmp["advancedplacement_courses_contain_Calculus"]=="No"]["% Math 4s target"],
                equal_var = False))
high_school_directory_merge["partner_cbo_cnt"]=high_school_directory_merge["partner_cbo"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
high_school_directory_merge["partner_hospital_cnt"]=high_school_directory_merge["partner_hospital"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
high_school_directory_merge["partner_highered_cnt"]=high_school_directory_merge["partner_highered"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
high_school_directory_merge["partner_cultural_cnt"]=high_school_directory_merge["partner_cultural"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
high_school_directory_merge["partner_nonprofit_cnt"]=high_school_directory_merge["partner_nonprofit"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
high_school_directory_merge["partner_corporate_cnt"]=high_school_directory_merge["partner_corporate"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
high_school_directory_merge["partner_financial_cnt"]=high_school_directory_merge["partner_financial"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
high_school_directory_merge["partner_other_cnt"]=high_school_directory_merge["partner_other"].apply(lambda x:len(x.split(',')) if type(x)==str else 0)
tmp=high_school_directory_merge[["% ELA 4s target","% Math 4s target","partner_cbo_cnt","partner_hospital_cnt","partner_highered_cnt","partner_cultural_cnt",
         "partner_nonprofit_cnt","partner_corporate_cnt","partner_financial_cnt","partner_other_cnt"]]

corrmat = tmp.corr()
plt.figure(figsize=(16,6))
ax=plt.subplot(121)
sns.heatmap(corrmat, square=True, linewidths=.5, annot=True,cmap=plt.cm.get_cmap('RdYlBu'))
plt.title("Correation Matrix",size=15)

plt.subplot(122)
high_school_directory_merge["extracurricular_activities_art_cnt"]=high_school_directory_merge["extracurricular_activities"].apply(lambda x:x.lower().count("art")+x.lower().count("cultur") if type(x)==str else 0)
sns.pointplot(x=high_school_directory_merge["extracurricular_activities_art_cnt"],y=high_school_directory_merge["% ELA 4s target"],color="red",jitter=False)
plt.title("# of Cultural/Art organizations vs ELA Excellent Rate",size=15)
plt.show()
