import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

#We're loading in the Multiple Choice answers as our main data set.
multiple_choice_df = pd.read_csv("../input/multipleChoiceResponses.csv", low_memory=False)


no_deg_ans = ['No formal education past high school',
              'Some college/university study without earning a bachelor’s degree']
pref_not_ans = ['I prefer not to answer']

no_deg_career_excl = ['Student', np.nan]

deg_career_excl = [np.nan]

no_deg = multiple_choice_df.loc[(multiple_choice_df['Q4'].isin(no_deg_ans)) &
                                (~multiple_choice_df['Q6'].isin(no_deg_career_excl))]
deg = multiple_choice_df.loc[(~multiple_choice_df.index.isin(no_deg.index)) &
                             (~multiple_choice_df['Q4'].isin(pref_not_ans)) &
                             (~multiple_choice_df['Q6'].isin(deg_career_excl))]

no_deg_total = no_deg.shape[0]
deg_total = deg.shape[0]
omitted = multiple_choice_df.shape[0] - no_deg_total - deg_total - 1
print ('No Degree:', no_deg_total)
print ('Degree or Attaining:', deg_total)

pie_fig, pie_ax = plt.subplots()
pie_ax.pie([no_deg_total,deg_total, omitted],explode = [0,0.1,0.0],
           labels=['No Degree', 'Degree or Attaining', 'Omitted'],
          autopct='%2.1f%%', shadow=True, startangle=90,
           labeldistance=1.25, pctdistance=1.1)
pie_ax.axis('equal')
plt.show()

def hist_dict(df,question,supress_nan=False,title=None):
    careers = defaultdict(int)
    for val in df[question].values:
        #This massive block is just shortening some of the answers for plotting sake
        try:
            if val.find('retired') > 0:
                val = 'Retired Option'
            elif val == 'I have never written code but I want to learn':
                val = 'Never written code but want to learn'
            elif val == 'I have never written code and I do not want to learn':
                val = "Never written code and don't want to learn"
            elif val == 'I have never studied machine learning but plan to learn in the future':
                val = 'Never studied ML, but plan to in the future'
            elif val == 'I have never studied machine learning and I do not plan to':
                val = "Never studied ML, and don't plan to"
            elif val == 'Dataset aggregator/platform (Socrata, Kaggle Public Datasets Platform, etc.)':
                val = 'Dataset aggregator/platform'
            elif val == 'Independent projects are much more important than academic achievements':
                val = 'Independent projects are much more important'
            elif val == 'Independent projects are slightly more important than academic achievements':
                val = 'Independent projects are slighty more important'
            elif val == 'Independent projects are much less important than academic achievements':
                val = 'Independent projects are much less important'
            elif val == 'Independent projects are slightly less important than academic achievements':
                val = 'Independent projects are slightly less important'
            elif val == 'Independent projects are equally important as academic achievements':
                val = 'Independent projects are equally as important'
            elif val == 'United Kingdom of Great Britain and Northern Ireland':
                val = 'UK and Northern Ireland'
        except:
            pass
        try:
            if np.isnan(val):
                if supress_nan:
                    continue
                val = 'Not Answered'
        except:
            pass
        careers[val] += 1
    #Note if there is not a title passed, it simply returns the defaultdict of frequencies
    if title is None:
        return dict(careers)
    
    fig = plt.figure()
    fig.set_size_inches(12,6)
    ax = fig.add_subplot(111)
    
    #Here we're ranking the bars in descending order
    careers = sorted(careers.items(), key = lambda kv: kv[1], reverse=True)
    career_names,totals = zip(*careers)
    index = range(len(careers))
    
    ax.bar(index,totals)
    ax.set_xticks(index)
    ax.set_xticklabels(career_names,rotation='vertical')
    ax.set_title(title)
    return dict(careers)
def all_that_apply_hist(df,question):
    q = []
    for col in df.columns:
        if col.find(question) >= 0:
            q.append(col)
    #We ignore the column that gives us the location of the other text
    q = [x for x in q if x != '{}_OTHER_TEXT'.format(question)]
    q_df = df[q]
    
    #this is the totla number of answers given by each response
    row_sum = q_df.notnull().sum(axis=1)
    col_counts = defaultdict(int)
    for col in q_df.columns:
        vals = q_df[col].unique()
        if len(vals) < 2:
            continue
        column = [val for val in vals if not pd.isnull(val)][0]
        #This block is to shorten very long answers for plotting purposes
        if column == 'Dataset aggregator/platform (Socrata, Kaggle Public Datasets Platform, etc.)':
                column = 'Dataset aggregator/platform'
        #the total number of occurances for each option
        col_counts[column] += q_df[col].notnull().sum(axis=0)
    return col_counts, row_sum.values
def add_to_100_hist(df,question):
    means = defaultdict(float)
    for col in df.columns:
        if col.find(question) >= 0:
            if col.find('OTHER_TEXT') >= 0:
                continue
            if question == 'Q35':
                if col == 'Q35_Part_1':
                    means['Self-taught'] += df[col].astype(float).mean()
                elif col == 'Q35_Part_2':
                    means['Online Courses'] += df[col].astype(float).mean()
                elif col == 'Q35_Part_3':
                    means['Work'] += df[col].astype(float).mean()
                elif col == 'Q35_Part_4':
                    means['University'] += df[col].astype(float).mean()
                elif col == 'Q35_Part_5':
                    means['Kaggle Competitions'] += df[col].astype(float).mean()
                elif col == 'Q35_Part_6':
                    means['Other'] += df[col].astype(float).mean()
            if question == 'Q34':
                if col == 'Q34_Part_1':
                    means['Gathering Data'] += df[col].astype(float).mean()
                elif col == 'Q34_Part_2':
                    means['Cleaning Data'] += df[col].astype(float).mean()
                elif col == 'Q34_Part_3':
                    means['Visualizing Data'] += df[col].astype(float).mean()
                elif col == 'Q34_Part_4':
                    means['Model Building/Model Selection'] += df[col].astype(float).mean()
                elif col == 'Q34_Part_5':
                    means['Putting the model into production'] += df[col].astype(float).mean()
                elif col == 'Q34_Part_6':
                    means['Finding insights/communicating with stakeholders'] += df[col].astype(float).mean()
    return means
def compare_bar_plot(data1,total1,label1,data2,total2,label2,title,y_label=None,only_both=True,width=0.35):
    if only_both:
        remove = []
        for name in data1.keys():
            if name not in data2.keys():
                remove.append(name)
        for name in data2.keys():
            if name not in data1.keys():
                remove.append(name)
        remove = list(set(remove))
        for key in remove:
            if key in data1:
                del data1[key]
            if key in data2:
                del data2[key]
                
    if len(data1) < len(data2):
        data1, data2 = data2, data1
        total1, total2 = total2 , total1
        label1, label2 = label2, label1
    data1 = sorted(data1.items(),key = lambda kv: kv[1], reverse=True)
    names, vals1 = zip(*data1)
    index = np.arange(len(names))
    vals1 = [x/total1 for x in vals1]
    vals2 = []
    for name in names:
        if name not in data2.keys():
            vals2.append(0)
        else:
            vals2.append(data2[name]/total2)
    for name in data2.keys():
        if name not in names:
            vals1.append(0)
            vals2.append(data2[name]/total2)
    fig = plt.figure()
    fig.set_size_inches(12,7)
    ax = fig.add_subplot(111)
    rects1 = ax.bar(index, vals1,width, label=label1)
    rects2 = ax.bar(index + width, vals2,width, label=label2)
    if y_label is None:
        ax.set_ylabel('Fraction of Kagglers')
    else:
        ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.set_xticks(index + width / 2)
    ax.set_xticklabels(names, rotation='vertical')
    ax.legend()
    plt.show()
def component_prop_test(counts_dict1,total1,counts_dict2,total2,title,sig_level=0.05,full_print=False):
    def prop_test(deg_count,deg_total,no_deg_count,no_deg_total):
        arr = np.array([[deg_count,deg_total-deg_count], 
                        [no_deg_count, no_deg_total-no_deg_count]])
        chi2, p, dof, exp = stats.chi2_contingency(arr)
        return chi2, p
    
    longest = len(title)
    for key in counts_dict1:
        if len(key) > longest:
            longest = len(key)
    for key in counts_dict2:
        if len(key) > longest:
            longest = len(key)
            
    print (('{:>' + str(longest) + '}     {:>9}     {:>20}').format(title, 'P Value', 'Appeared Significance'))
    print ('-'*min([longest+47,92]))
    for key, val in counts_dict1.items():
        if key not in counts_dict2.keys():
            if full_print:
                print ('**{} only recorded with a Degree: {}**'.format(title,key))
            continue
        if type(total1) == dict:
            chi2, p = prop_test(val,total1[key],counts_dict2[key],total2[key])
            if p < sig_level:
                if val/total1[key] < counts_dict2[key]/total2[key]:
                    sig = 'More likely without a Degree'
                else:
                    sig = 'More likely with a Degree'
            else:
                sig = 'No'
            print (('{:>' + str(longest) + '}     {:.3e}     {:>20}').format(key,p,sig))
        else:
            chi2, p = prop_test(val,total1,counts_dict2[key], total2)
            if p < sig_level:
                if val/total1 < counts_dict2[key]/total2:
                    sig = 'More likely without a Degree'
                else:
                    sig = 'More likely with a Degree'
            else:
                sig = 'No'
            print (('{:>' + str(longest) + '}     {:.3e}     {:>20}').format(key,p,sig))
    for key, val in counts_dict2.items():
        if key not in counts_dict1.keys():
            if full_print:
                print ('**{} only recorded without a Degree: {}**'.format(title,key))
def t_test(a,a_label,b,b_label,title,sig_val=0.05):
    t_stat, p = stats.ttest_ind(a,b)
    print ('Mean {} of {}: {:1.2}'.format(title,a_label, np.mean(a)))
    print ('Mean {} of {}: {:1.2}'.format(title,b_label, np.mean(b)))
    print ('\n')
    print (('{:>' + str(len(title)) + '}     {:>6}    {:>12}').format('Mean ' + title,'P value','Significant Difference'))
    print ('-'*min([len(title)+43,92]))
    if p > sig_val:
        val = 'No'
    else:
        if np.mean(a) > np.mean(b):
            val = 'More with {}'.format(a_label)
        else:
            val = 'More with {}'.format(b_label)
    print (('{:>' + str(len(title)) + '}    {:.3e}    {:>12}').format('Mean ' + title,p,val))
def chi2(dist1,dist2,title,sig_level = 0.05):
    dist1 = list(dist1.values())
    dist2 = list(dist2.values())
    cont = np.array([dist1,dist2])
    chi2, p, df, exp = stats.chi2_contingency(cont)
    if p < sig_level:
        val = 'Yes'
    else:
        val = 'No'
    print (('{:<' + str(len(title)) + '}     {:>9}     {:>12}').format(title,'P Value','Appeared Significance'))
    print ('-'*min([len(title)+42,92]))
    print (('{:<' + str(len(title)) + '}     {:.3e}     {:>12}').format(title, p, val))
no_degree_career_counts = hist_dict(no_deg,'Q6',title='Without a Degree')
degree_career_counts = hist_dict(deg,'Q6',title='With or Attaining a Degree')
deg = deg.loc[~deg['Q6'].isin(no_deg_career_excl)]
old_deg_total = deg_total
deg_total = deg.shape[0]
print ('No Degree:', no_deg_total)
print ('Degree:', deg_total)
excluded_students = old_deg_total - deg_total

pie_fig, pie_ax = plt.subplots()
pie_ax.pie([no_deg_total,deg_total, omitted,excluded_students],
           explode = [0,0.1,0.0,0.0],
           labels=['No Degree', 'Degree', 'Omitted','Students'],
           autopct='%2.1f%%', shadow=True, startangle=90,
           labeldistance=1.3, pctdistance=1.175)
pie_ax.axis('equal')
plt.show()

degree_career_counts = hist_dict(deg,'Q6',title='With a Degree')
compare_bar_plot(degree_career_counts,deg_total,'Degree',
                 no_degree_career_counts,no_deg_total,'No Degree',
                'Comparing the Fraction in Different Positions')

component_prop_test(degree_career_counts,deg_total,
                    no_degree_career_counts,no_deg_total,
                    'Career')

no_deg_unemployment = 100*(no_degree_career_counts['Not employed']/no_deg_total)
deg_unemployment = 100*(degree_career_counts['Not employed']/deg_total)
print ('No Degree Unemployment Rate: {:.3}%'.format(no_deg_unemployment))
print ('Degree Unemployment Rate: {:.3}%'.format(deg_unemployment))
def unemployment_per_country(unemployed_dict,total_dict):
    rates = {}
    for key, val in unemployed_dict.items():
        rates[key] = val/total_dict[key]
    return rates

unemployed_deg = deg.loc[deg['Q6'] == 'Not employed']
unemployed_no_deg = no_deg.loc[no_deg['Q6'] == 'Not employed']

deg_country_counts = hist_dict(deg,'Q3')
no_deg_country_counts = hist_dict(no_deg,'Q3')

unemployed_deg_country = hist_dict(unemployed_deg,'Q3')
unemployed_no_deg_country = hist_dict(unemployed_no_deg,'Q3')

unemployed_deg_rates = unemployment_per_country(unemployed_deg_country,deg_country_counts)
unemployed_no_deg_rates = unemployment_per_country(unemployed_no_deg_country,no_deg_country_counts)

compare_bar_plot(unemployed_deg_rates,1,'Degree',
                 unemployed_no_deg_rates,1, 'No Degree',
                'Comparing Unemployment Rate by Country',y_label='Unemployment Rate')

component_prop_test(unemployed_deg_country,deg_country_counts,
                   unemployed_no_deg_country,no_deg_country_counts,
                   'Unemployment Rates')
deg_age_counts = hist_dict(unemployed_deg,'Q2')
no_deg_age_counts = hist_dict(unemployed_no_deg, 'Q2')

compare_bar_plot(deg_age_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_age_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing the Unemployment Age')

component_prop_test(deg_age_counts,unemployed_deg.shape[0],
                    no_deg_age_counts,unemployed_no_deg.shape[0],
                    'Age Range')
deg_lang_counts, deg_lang_totals = all_that_apply_hist(unemployed_deg,'Q16')
no_deg_lang_counts, no_deg_lang_totals = all_that_apply_hist(unemployed_no_deg,'Q16')

t_test(deg_lang_totals,'Degree',no_deg_lang_totals,'No Degree','Number of Languages')

compare_bar_plot(deg_lang_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_lang_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Languages', only_both=False)

component_prop_test(deg_lang_counts,unemployed_deg.shape[0],
                    no_deg_lang_counts,unemployed_no_deg.shape[0],
                    'Languages')
deg_primlang_counts = hist_dict(unemployed_deg,'Q17', supress_nan=True)
no_deg_primlang_counts = hist_dict(unemployed_no_deg, 'Q17',supress_nan=True)

compare_bar_plot(deg_primlang_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_primlang_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Primary Language Choice', only_both=False)

component_prop_test(deg_primlang_counts,unemployed_deg.shape[0],
                    no_deg_primlang_counts,unemployed_no_deg.shape[0],
                    'Primary Language')
deg_ml_counts, deg_ml_totals = all_that_apply_hist(unemployed_deg,'Q19')
no_deg_ml_counts, no_deg_ml_totals = all_that_apply_hist(unemployed_no_deg,'Q19')

t_test(deg_ml_totals,'Degree',no_deg_ml_totals,'No Degree','Machine Learning Frameworks')


compare_bar_plot(deg_ml_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_ml_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Maching Learning Frameworks', only_both=False)

component_prop_test(deg_ml_counts,unemployed_deg.shape[0],
                    no_deg_ml_counts,unemployed_no_deg.shape[0],
                    'Machine Learning Frameworks')
deg_primml_counts = hist_dict(unemployed_deg,'Q20', supress_nan=True)
no_deg_primml_counts = hist_dict(unemployed_no_deg, 'Q20',supress_nan=True)

compare_bar_plot(deg_primml_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_primml_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Primary ML Framework Choice', only_both=False)

component_prop_test(deg_primml_counts,unemployed_deg.shape[0],
                    no_deg_primml_counts,unemployed_no_deg.shape[0],
                    'Primary ML Framework')
deg_exp_counts = hist_dict(unemployed_deg,'Q24', supress_nan=True)
no_deg_exp_counts = hist_dict(unemployed_no_deg, 'Q24',supress_nan=True)

compare_bar_plot(deg_exp_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_exp_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Coding with Data Experience')

component_prop_test(deg_exp_counts,unemployed_deg.shape[0],
                    no_deg_exp_counts,unemployed_no_deg.shape[0],
                    'Coding Experience')
deg_mlexp_counts = hist_dict(unemployed_deg,'Q25', supress_nan=True)
no_deg_mlexp_counts = hist_dict(unemployed_no_deg, 'Q25',supress_nan=True)

compare_bar_plot(deg_mlexp_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_mlexp_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Machine Learning Experience')

component_prop_test(deg_mlexp_counts,unemployed_deg.shape[0],
                    no_deg_mlexp_counts,unemployed_no_deg.shape[0],
                    'Machine Learning Experience')
deg_ds_counts = hist_dict(unemployed_deg,'Q26', supress_nan=True)
no_deg_ds_counts = hist_dict(unemployed_no_deg, 'Q26',supress_nan=True)

compare_bar_plot(deg_ds_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_ds_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Self Evaluation')

component_prop_test(deg_ds_counts,unemployed_deg.shape[0],
                    no_deg_ds_counts,unemployed_no_deg.shape[0],
                    'Self Evalulation')
deg_data_counts, deg_data_totals = all_that_apply_hist(unemployed_deg,'Q31')
no_deg_data_counts, no_deg_data_totals = all_that_apply_hist(unemployed_no_deg,'Q31')

t_test(deg_data_totals,'Degree',no_deg_data_totals,'No Degree','Number of Types of Data')

compare_bar_plot(deg_data_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_data_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Type of Data')

component_prop_test(deg_data_counts,unemployed_deg.shape[0],
                    no_deg_data_counts,unemployed_no_deg.shape[0],
                    'Types of Data')
deg_primdata_counts = hist_dict(unemployed_deg,'Q32', supress_nan=True)
no_deg_primdata_counts = hist_dict(unemployed_no_deg, 'Q32',supress_nan=True)

compare_bar_plot(deg_primdata_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_primdata_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Primary Data Type')

component_prop_test(deg_primdata_counts,unemployed_deg.shape[0],
                    no_deg_primdata_counts,unemployed_no_deg.shape[0],
                    'Primary Data Type')
deg_source_counts, deg_source_totals = all_that_apply_hist(unemployed_deg,'Q33')
no_deg_source_counts, no_deg_source_totals = all_that_apply_hist(unemployed_no_deg,'Q33')

t_test(deg_source_totals,'Degree',no_deg_source_totals,'No Degree',' Number of Sources of Data')

compare_bar_plot(deg_source_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_source_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Sources of Data')

component_prop_test(deg_source_counts,unemployed_deg.shape[0],
                    no_deg_source_counts,unemployed_no_deg.shape[0],
                    'Source of Data')
deg_project = add_to_100_hist(unemployed_deg,'Q34')
no_deg_project = add_to_100_hist(unemployed_no_deg,'Q34')

compare_bar_plot(deg_project,100,'Degree',
                 no_deg_project,100, 'No Degree', 
                 'Average Project Distributions', y_label='Fraction of Time Spent')

chi2(deg_project,no_deg_project,'Project Distribtution')
deg_learning = add_to_100_hist(unemployed_deg,'Q35')
no_deg_learning = add_to_100_hist(unemployed_no_deg,'Q35')

compare_bar_plot(deg_learning,100,'Degree',
                 no_deg_learning,100, 'No Degree', 
                 'Average Learning Distributions',y_label='Fraction of Learning')

chi2(deg_learning,no_deg_learning,'Learning Distribtution')
deg_course_counts, deg_course_totals = all_that_apply_hist(unemployed_deg,'Q36')
no_deg_course_counts, no_deg_course_totals = all_that_apply_hist(unemployed_no_deg,'Q36')

t_test(deg_course_totals,'Degree',no_deg_course_totals,'No Degree','Online Courses')

compare_bar_plot(deg_course_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_course_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Online Courses')

component_prop_test(deg_course_counts,unemployed_deg.shape[0],
                    no_deg_course_counts,unemployed_no_deg.shape[0],
                    'Online Courses')
deg_primcourse_counts = hist_dict(unemployed_deg,'Q37', supress_nan=True)
no_deg_primcourse_counts = hist_dict(unemployed_no_deg, 'Q37',supress_nan=True)

compare_bar_plot(deg_primcourse_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_primcourse_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Primary Online Course')

component_prop_test(deg_primcourse_counts,unemployed_deg.shape[0],
                    no_deg_primcourse_counts,unemployed_no_deg.shape[0],
                    'Primary Online Course')
deg_media_counts, deg_media_totals = all_that_apply_hist(unemployed_deg,'Q38')
no_deg_media_counts, no_deg_media_totals = all_that_apply_hist(unemployed_no_deg,'Q38')

t_test(deg_media_totals,'Degree',no_deg_media_totals,'No Degree','Media Sources')

compare_bar_plot(deg_media_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_media_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Media Sourses', only_both=False)

component_prop_test(deg_media_counts,unemployed_deg.shape[0],
                    no_deg_media_counts,unemployed_no_deg.shape[0],
                    'Media Sourses')
deg_online_counts = hist_dict(unemployed_deg,'Q39_Part_1', supress_nan=True)
no_deg_online_counts = hist_dict(unemployed_no_deg, 'Q39_Part_1',supress_nan=True)

compare_bar_plot(deg_online_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_online_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Opinion of Online Education vs Traditional')

component_prop_test(deg_online_counts,unemployed_deg.shape[0],
                    no_deg_online_counts,unemployed_no_deg.shape[0],
                    'Opinion of Online Education vs Traditional')
deg_bootcamp_counts = hist_dict(unemployed_deg,'Q39_Part_2', supress_nan=True)
no_deg_bootcamp_counts = hist_dict(unemployed_no_deg, 'Q39_Part_2',supress_nan=True)

compare_bar_plot(deg_bootcamp_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_bootcamp_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Opinion of Bootcamp Education vs Traditional')

component_prop_test(deg_bootcamp_counts,unemployed_deg.shape[0],
                    no_deg_bootcamp_counts,unemployed_no_deg.shape[0],
                    'Opinion of Bootcamp Education vs Traditional')
deg_ind_counts = hist_dict(unemployed_deg,'Q40', supress_nan=True)
no_deg_ind_counts = hist_dict(unemployed_no_deg, 'Q40',supress_nan=True)

compare_bar_plot(deg_ind_counts,unemployed_deg.shape[0],'Degree',
                 no_deg_ind_counts,unemployed_no_deg.shape[0], 'No Degree',
                'Comparing Independent Projects vs Academic Achievements')

component_prop_test(deg_ind_counts,unemployed_deg.shape[0],
                    no_deg_ind_counts,unemployed_no_deg.shape[0],
                    'Independent Projects vs Academic Achievements')