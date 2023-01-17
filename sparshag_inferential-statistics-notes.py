import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st
import seaborn as sns
#Generate 25-States list and randomly take a sample of 8
states = ["s" + str(i) for i in range(1,26)]
np.random.choice(states, size=8, replace=False, p=None)
#Import Employee File and Create Stratefied Samples for each Job-level and Department
emp = pd.read_csv("../input/emp.csv")
display(emp)

tab_emp = pd.crosstab(index = emp["Dept"],  columns=emp["Job_Level"], colnames = ['']) 
display(tab_emp)

emp.groupby(['Dept','Job_Level'], group_keys=False).apply(lambda x: x.sample(min(len(x), 1)))
#Import Student Marks Sample and Draw Histogram along with calculating Sample Mean, S.d. and Variance
student_mark = pd.read_csv("../input/students_mark.csv")
display(student_mark.head())

#Histogram of Sample
plt.hist(student_mark.Mark,bins=range(160, 620, 20),edgecolor='black', linewidth=1)
plt.title("Distribution of Marks")
plt.xlabel("Marks")
plt.ylabel("Frequency")
plt.show()

#Mean, Variance and S.d. of Sample
sample_mean = np.mean(student_mark.Mark)
sample_variance = np.var(student_mark.Mark)
sample_sd = np.std(student_mark.Mark)
print(sample_mean,sample_variance,sample_sd)
meanp = 10.45; sdp = 2.5
cl = .90; E=1
sample_size = np.ceil((1.645*sdp/E)**2)
print('required sample size is',sample_size)
meanp = 72
sdp = 12.5
sample_size = 10
sderr = sdp/np.sqrt(sample_size)
zscore = (80-72)/sderr #since we know the population s.d., we will calculate z-probability, not t.
print('probability of mean score > 80 is',(1-st.norm.cdf(tscore))*100,'%')
#Distribution of Outcomes
dice_output = pd.read_excel("../input/dice.xlsx",sheet_name=0, header=0)
tab_dice_output = pd.crosstab(index = dice_output.Output,  columns="count", colnames = [''])
tab_dice_output.plot.bar()
plt.show()


#Sample size 3 into dataframe
sample_3 = []
for i in range(1,1001):
    x = np.random.choice(dice_output.Output, size=3, replace=True).tolist()
    sample_3.append(x)
    
df_sample_3 = pd.DataFrame(sample_3).T
df_sample_3.shape


#Sample Size 3  - Plot Sampling distribution of sample mean
sample_mean_3 = df_sample_3.apply(np.mean, axis=0)
sample_mean_3.describe()

dice_population_mean = np.mean(dice_output.Output)
dice_population_sd = np.std(dice_output.Output)

plt.hist(sample_mean_3,bins=np.arange(0.0, 7.0, 0.25),edgecolor='black', linewidth=1)
plt.title("Sampling Distribution of mean for n=3")
plt.xlabel("Range")
plt.ylabel("Frequency")
plt.show()

mean_of_sample_mean_3 = np.mean(sample_mean_3)
print(mean_of_sample_mean_3,dice_population_mean)

std_of_sample_mean_3 = np.std(sample_mean_3)
print(std_of_sample_mean_3,dice_population_sd)


#Sample Size 3  - Plot Sampling distribution of sample mean
sample_10 = []
for i in range(1,1001):
    x = np.random.choice(dice_output.Output, size=10, replace=True).tolist()
    sample_10.append(x)
    
df_sample_10 = pd.DataFrame(sample_10).T

sample_mean_10 = df_sample_10.apply(np.mean, axis=0)
sample_mean_10.describe()

dice_population_mean = np.mean(dice_output.Output)
dice_population_sd = np.std(dice_output.Output)

plt.hist(sample_mean_10,bins=np.arange(0.0, 7.0, 0.25),edgecolor='black', linewidth=1)
plt.title("Sampling Distribution of mean for n=10")
plt.xlabel("Range")
plt.ylabel("Frequency")
plt.show()

mean_of_sample_mean_10 = np.mean(sample_mean_10)
print(mean_of_sample_mean_10,dice_population_mean)

std_of_sample_mean_10 = np.std(sample_mean_10)
print(std_of_sample_mean_10,dice_population_sd)
meanp = 473
sdp = 3
sample_size = 8
sderr = sdp/np.sqrt(8)
zscore1 = (470-meanp)/sderr
zscore2 = (475-meanp)/sderr
print('probability that the mean is less than 470 is',(st.norm.cdf(zscore1))*100,'%')
print('probability that the mean is more than 475 is',(1-st.norm.cdf(zscore2))*100,'%')
meanp = 2
sdp = 0.7
sample_size = 50
test_limit = 110/50
sderr = sdp/np.sqrt(sample_size)
zscore = (test_limit-meanp)/sderr
print('probability of running out of water (assuming they drink equal amount of water) is',(1-st.norm.cdf(zscore))*100,'%')
z_crit = 1.96 #z-score at 95% CL
sd = 85.35
E = 5
sample_size = np.ceil((z_crit*sd/E)**2)
print('Calculated Sample size is',sample_size)
student = pd.read_csv('../input/students_mark.csv')
student.head()
#the distribution of marks
variable = student['Mark']
sns.distplot(variable)
plt.title('The data looks like Normally Distributed (Assumption)')

mean = np.mean(variable)
var = np.var(variable)
std = np.std(variable)
stderr = (std/np.sqrt(sample_size))
ci_low = mean - stderr*st.norm._ppf((1+0.95)/2.)
ci_high = mean + stderr*st.norm._ppf((1+0.95)/2.)
print('Sampling distribution of mean = ',mean)
print('Standard error = ',stderr)
print('Confidence Interval',ci_low,ci_high)
var_low = var*(sample_size-1)/(np.round(st.chi2.ppf((1+0.95)/2.,sample_size-1),2))
var_high = var*(sample_size-1)/(np.round(st.chi2.ppf((1-0.95)/2.,sample_size-1),2))
sample_mean = 403
sample_sd = 75
sample_size = 100
population_mean = 413
significance_level = 0.05
standard_error = sample_sd/np.sqrt(sample_size)

z_score = (sample_mean - population_mean)/standard_error
p_value = st.norm.cdf(z_score)
print('Z-socre is ',z_score,' and P-value is ',p_value*100,'%')
print('P-value is more than significance level, So we do not reject Null Hypothesis.')
school1 = student.loc[student['SchoolID']=='ST1S1','Mark']
school2 = student.loc[student['SchoolID']=='ST1S3','Mark']

mean_s1 = np.mean(school1); sd_s1 = np.std(school1)
mean_s2 = np.mean(school2); sd_s2 = np.std(school2)
print('Mean and s.d. of School 1:',mean_s1,',',sd_s1,'\nMean and s.d. of School 2:',mean_s2,',',sd_s2)
#Point estimate of the difference in population mean -> as sample is unbiased estimator of population mean....
mean_diff = abs(mean_s1 - mean_s2)
# Side-question on independent sample t-test
## The average battery life of two smartphones from two manufacturers are compared. 
## 10 samples are picked from each of the smart phones and the average battery life 
## was found to be 400 minutes and 480 minutes having standard deviation as 30 minutes 
## and 26 minutes respectively. Determine the confidence interval for the difference in 
## the population mean with 0.95 confidence level.

meanx = 80
fx = (30*30)/(26*26) #1.33 - so go with pooled s.d.
sdpx = np.sqrt(((9*30*30)+(9*26*26))/(18))
sdperrx = sdpx * np.sqrt((1/10)+(1/10))
cilowx = meanx - st.t.ppf((1+0.95)/2,18)*sdperrx
cihighx = meanx + st.t.ppf((1+0.95)/2,18)*sdperrx

print('\nConfidence Interval is (%s, %s)'%(cilowx,cihighx))
print('Variance ratio of the 2 samples: ',np.var(school1)/np.var(school2), 'which is close to 1, so we will use pooled s.d.')
pooled_sd = np.sqrt((((len(school1)-1)*np.var(school1, ddof=1))+((len(school2)-1)*np.var(school2, ddof=1)))/(len(school1)+len(school2)-2))
print('Pooled s.d. is',pooled_sd)
stderr_diff = pooled_sd * np.sqrt((1/len(school1))+(1/len(school2)))
print('Calculated standard error is',stderr_diff)
cilow_diff = mean_diff - st.t.ppf((1+0.95)/2,38)*stderr_diff
cihigh_diff = mean_diff + st.t.ppf((1+0.95)/2,38)*stderr_diff

print('\nConfidence Interval is (%s, %s)'%(cilow_diff,cihigh_diff))
#t-statistic
t_score = (mean_s1 - mean_s2)/stderr_diff #0.9995
t_critical = st.t.ppf((1+0.95)/2,38)

print('t-statistic %s is lower than t-critical %s, so we failed to reject the Null hypothesis'%(t_score,t_critical))
teacher = pd.read_csv('Mark_teacher_gender.csv')
teacher.head(3)
sns.boxplot(x='Teacher', y='Mark', data=teacher)
#Overall stats
mean_overall = np.mean(teacher['Mark'])
var_overall = np.var(teacher['Mark'], ddof=1)
sd_overall = np.std(teacher['Mark'], ddof=1)
n_all = len(teacher)

#stats for teacher 1
mean_t1 = np.mean(teacher.loc[teacher['Teacher']=='teacher1','Mark'])
var_t1 = np.var(teacher.loc[teacher['Teacher']=='teacher1','Mark'], ddof=1)
sd_t1 = np.std(teacher.loc[teacher['Teacher']=='teacher1','Mark'], ddof=1)
df_t1 = teacher[teacher['Teacher']=='teacher1']
n_t1 = len(df_t1)

#stats for teacher 2
mean_t2 = np.mean(teacher.loc[teacher['Teacher']=='teacher2','Mark'])
var_t2 = np.var(teacher.loc[teacher['Teacher']=='teacher2','Mark'], ddof=1)
sd_t2 = np.std(teacher.loc[teacher['Teacher']=='teacher2','Mark'], ddof=1)
df_t2 = teacher[teacher['Teacher']=='teacher2']
n_t2 = len(df_t2)

#stats for teacher 3
mean_t3 = np.mean(teacher.loc[teacher['Teacher']=='teacher3','Mark'])
var_t3 = np.var(teacher.loc[teacher['Teacher']=='teacher3','Mark'], ddof=1)
sd_t3 = np.std(teacher.loc[teacher['Teacher']=='teacher3','Mark'], ddof=1)
df_t3 = teacher[teacher['Teacher']=='teacher3']
n_t3 = len(df_t3)
sst = sum((teacher['Mark']-mean_overall)**2)
ssw = sum((df_t1.Mark - mean_t1)**2) + sum((df_t2.Mark - mean_t2)**2) + sum((df_t3.Mark - mean_t3)**2)
ssb = n_t1*((mean_t1 - mean_overall)**2) + n_t2*((mean_t2 - mean_overall)**2) + n_t3*((mean_t3 - mean_overall)**2)
dof_sst = n_all - 1
dof_ssb = 2
dof_ssw = 3*(6-1) #N-k

mst = sst/dof_sst
msb = ssb/dof_ssb
msw = ssw/dof_ssw
f_calculated = msb/msw
f_critical = st.f.ppf(q=0.95, dfn=dof_ssb, dfd=dof_ssw)
print('F-calculated(%s) is more than F-critical(%s), so we reject Null Hypothesis.\
      \nThis implies that the pattern we observed prima facie in box plot is actually True'%(f_calculated,f_critical))
# Directly calculating f-statistic using scipy function
result_anova = st.f_oneway(teacher.loc[teacher.Teacher == "teacher1","Mark"], teacher.loc[teacher.Teacher == "teacher2","Mark"]
                           ,teacher.loc[teacher.Teacher == "teacher3","Mark"])
print(result_anova)
mark_scored = teacher.copy()
#Overall Mean 
overall_mean = np.mean(mark_scored.Mark)
print("overall_mean:",overall_mean)

#Means with respect to teacher
mean_teacher1 = np.mean(mark_scored.loc[mark_scored.Teacher == "teacher1","Mark"])
print("mean_teacher1:",mean_teacher1)
mean_teacher2 = np.mean(mark_scored.loc[mark_scored.Teacher == "teacher2","Mark"])
print("mean_teacher2:",mean_teacher2)
mean_teacher3 = np.mean(mark_scored.loc[mark_scored.Teacher == "teacher3","Mark"])
print("mean_teacher3:",mean_teacher3)

#Means with respect to gender
mean_male = np.mean(mark_scored.loc[mark_scored.Gender == "Male","Mark"])
print("mean_male:",mean_male)
mean_female = np.mean(mark_scored.loc[mark_scored.Gender == "Female","Mark"])
print("mean_female",mean_female)

#Means with respect to combination of teacher and gender 
mean_male_teacher1 = np.mean(mark_scored.loc[(mark_scored.Teacher == "teacher1") & (mark_scored.Gender == "Male"),"Mark"])
print("mean_male_teacher1:",mean_male_teacher1)
mean_female_teacher1 = np.mean(mark_scored.loc[(mark_scored.Teacher == "teacher1") & (mark_scored.Gender == "Female"),"Mark"])
print("mean_female_teacher1:",mean_female_teacher1)
mean_male_teacher2 = np.mean(mark_scored.loc[(mark_scored.Teacher == "teacher2") & (mark_scored.Gender == "Male"),"Mark"])
print("mean_male_teacher2:",mean_male_teacher2)
mean_female_teacher2 = np.mean(mark_scored.loc[(mark_scored.Teacher == "teacher2") & (mark_scored.Gender == "Female"),"Mark"])
print("mean_female_teacher2:",mean_female_teacher2)
mean_male_teacher3 = np.mean(mark_scored.loc[(mark_scored.Teacher == "teacher3") & (mark_scored.Gender == "Male"),"Mark"])
print("mean_male_teacher3:",mean_male_teacher3)
mean_female_teacher3 = np.mean(mark_scored.loc[(mark_scored.Teacher == "teacher3") & (mark_scored.Gender == "Female"),"Mark"])
print("mean_female_teacher3:",mean_female_teacher3)
SST = np.sum((mark_scored.Mark - overall_mean)**2)
print("SST:",SST)

#sum of squares between, due to teacher
SSB_teacher1 = len(mark_scored.loc[mark_scored.Teacher == "teacher1","Mark"]) *((mean_teacher1 - overall_mean)**2)
SSB_teacher2 = len(mark_scored.loc[mark_scored.Teacher == "teacher2","Mark"]) *((mean_teacher2 - overall_mean)**2)
SSB_teacher3 = len(mark_scored.loc[mark_scored.Teacher == "teacher3","Mark"]) *((mean_teacher3 - overall_mean)**2)
SSB_teacher = SSB_teacher1 + SSB_teacher2 + SSB_teacher3
print("SSB_teacher:",SSB_teacher)

#sum of squares between, due to gender
SSB_male = len(mark_scored.loc[mark_scored.Gender == "Male","Mark"]) *((mean_male - overall_mean)**2)
SSB_female = len(mark_scored.loc[mark_scored.Gender == "Female","Mark"]) *((mean_female - overall_mean)**2)
SSB_gender = SSB_male + SSB_female
print("SSB_gender:",SSB_gender)

#sum of squares within (SSW) i.e. error
SSW1 = np.sum((mark_scored.loc[(mark_scored.Teacher == "teacher1") & (mark_scored.Gender == "Male"),"Mark"] - mean_male_teacher1)**2)
print("SSW1:",SSW1)
SSW2 = np.sum((mark_scored.loc[(mark_scored.Teacher == "teacher1") & (mark_scored.Gender == "Female"),"Mark"]- mean_female_teacher1)**2)
print("SSW2:",SSW2)
SSW3 = np.sum((mark_scored.loc[(mark_scored.Teacher == "teacher2") & (mark_scored.Gender == "Male"),"Mark"] - mean_male_teacher2)**2)
print("SSW3:",SSW3)
SSW4 = np.sum((mark_scored.loc[(mark_scored.Teacher == "teacher2") & (mark_scored.Gender == "Female"),"Mark"]- mean_female_teacher2)**2)
print("SSW4:",SSW4)
SSW5 = np.sum((mark_scored.loc[(mark_scored.Teacher == "teacher3") & (mark_scored.Gender == "Male"),"Mark"] - mean_male_teacher3)**2)
print("SSW5:",SSW5)
SSW6 = np.sum((mark_scored.loc[(mark_scored.Teacher == "teacher3") & (mark_scored.Gender == "Female"),"Mark"]- mean_female_teacher3)**2)
print("SSW6:",SSW6)
SSW = SSW1 + SSW2 + SSW3 + SSW4 + SSW5 + SSW6 
print("--------------------------")
print("SSW:",SSW)

#sum of square combined (SSC)
SSC = SST - SSB_teacher - SSB_gender - SSW
SSB_combined = SSC
print("SSB_combined:",SSB_combined)
df_total = len(mark_scored.Mark) - 1
print("df_total:",df_total)

Cat_teach= pd.Series(pd.Categorical(mark_scored.Teacher))
df_between_teacher = len(Cat_teach.cat.categories) - 1
print("df_between_teacher:",df_between_teacher)

Cat_gender= pd.Series(pd.Categorical(mark_scored.Gender))
df_between_gender = len(Cat_gender.cat.categories) - 1
print("df_between_gender:",df_between_gender)

df_combined = df_between_teacher * df_between_gender
print("df_combined:",df_combined)

df_within = len(Cat_teach.cat.categories) * len(Cat_gender.cat.categories) * 2
print("df_within:",df_within)


#Calculating mean square values
MSB_teacher = SSB_teacher / df_between_teacher
MSB_gender = SSB_gender / df_between_gender
MSB_combined = SSB_combined / df_combined
MSW = SSW / df_within
print("MSB_teacher:",MSB_teacher)
print("MSB_gender:",MSB_gender)
print("MSB_combined:",MSB_combined)
print("MSW:",MSW)
#Calculating F statistic
F_teacher = MSB_teacher / MSW
F_gender = MSB_gender / MSW
F_combined = MSB_combined / MSW
print("F_teacher:",F_teacher)
print("F_gender:",F_gender)
print("F_combined:",F_combined)

#Calculating critical F statistics
F_critical_teacher = st.f.ppf(0.95,2,12)
F_critical_gender = st.f.ppf(0.95,1,12)
F_critical_combined = st.f.ppf(0.95,2,12)
print("F_critical_teacher:",F_critical_teacher)
print("F_critical_gender:",F_critical_gender)
print("F_critical_combined:",F_critical_combined)

print("\nTeachers and Gender have significant performance on Performance individually, but not significant effect together")
import statsmodels.api as sm
from statsmodels.formula.api import ols

result_two_way_anova = ols('Mark ~ Teacher + Gender + Teacher:Gender', data = mark_scored).fit()
#result_two_way_anova.summary()
aov_table = sm.stats.anova_lm(result_two_way_anova, typ=2) #change value of typ = 1 or 2 or 3 for different views
aov_table
section = pd.read_excel('../input/marks_section.xlsx')
section
result_two_way_anova = ols('Mark ~ Section', data = section).fit()
#result_two_way_anova.summary()
aov_table = sm.stats.anova_lm(result_two_way_anova, typ=2) #change value of typ = 1 or 2 or 3 for different views
display(aov_table)
print('p-value tells us that we do not have enough evidence to reject H0, so student performance do not depends on section.')
mark_hour = pd.read_csv('../input/mark_teacher_hour.csv')
mark_hour
#finding correlation
Mark = mark_hour.Mark
Hours = mark_hour.Hours_Per_Day
print("Correlation between Marks and Hours", Mark.corr(Hours))

#Linear-model (regression) plot
sns.lmplot(x='Hours_Per_Day',y='Mark', data=mark_hour)

#Model fitting
from statsmodels.stats.anova import anova_lm
model = ols('Mark ~ Hours_Per_Day', mark_hour).fit()
model.summary()
markhour_teacher = pd.read_csv('../input/mark_teacher_hour.csv')
markhour_section = pd.read_excel('../input/mark_hour_2.xlsx')

#ANCOVA on combined categorical & continuous variable
from statsmodels.stats.anova import anova_lm
model = ols('Mark ~ Hours_per_Day + C(Section, Treatment(reference="None"))', markhour_section).fit()
display(model.summary())

#Linear-model (regression) plot comparing affect of teachers and affect of sections
sns.lmplot(x='Hours_Per_Day',y='Mark', hue='Teacher', data=markhour_teacher)
sns.lmplot(x='Hours_per_Day',y='Mark', hue='Section', data=markhour_section)
