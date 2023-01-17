import pandas as pd
path = r'../input'
course_offerings = pd.read_csv(path+'/course_offerings.csv')
courses = pd.read_csv(path+'/courses.csv')
grade_distributions = pd.read_csv(path+'/grade_distributions.csv')
instructors = pd.read_csv(path+'/instructors.csv')
rooms = pd.read_csv(path+'/rooms.csv')
schedules = pd.read_csv(path+'/schedules.csv')
sections = pd.read_csv(path+'/sections.csv')
subject_memberships = pd.read_csv(path+'/subject_memberships.csv')
subjects = pd.read_csv(path+'/subjects.csv')
teachings = pd.read_csv(path+'/teachings.csv')
df = [course_offerings,courses,grade_distributions,instructors,rooms,
     schedules,sections,subject_memberships,subjects,teachings]
for i in df:
    print(i.columns)
def grade_avg(worst_results):
    worst_results['total_count'] = worst_results[['a_count', 'ab_count',
           'b_count', 'bc_count', 'c_count', 'd_count', 'f_count']].sum(axis = 1)
    worst_results['a_count'] = worst_results['a_count']/worst_results['total_count']
    worst_results['ab_count'] = worst_results['ab_count']/worst_results['total_count']
    worst_results['b_count'] = worst_results['b_count']/worst_results['total_count']
    worst_results['bc_count'] = worst_results['bc_count']/worst_results['total_count']
    worst_results['d_count'] = worst_results['d_count']/worst_results['total_count']
    worst_results['c_count'] = worst_results['c_count']/worst_results['total_count']
    worst_results['f_count'] = worst_results['f_count']/worst_results['total_count']
    return worst_results[worst_results['a_count'].notnull()]
worst_results = grade_distributions[['course_offering_uuid', 'section_number', 'a_count', 'ab_count',
       'b_count', 'bc_count', 'c_count', 'd_count', 'f_count']].copy()
worst_results = grade_avg(worst_results)
worst_20 = worst_results[worst_results['total_count']>10].sort_values(['f_count','d_count',
                                                           'c_count','b_count'],ascending=False).head(20)
worst_20
# we need to remove these 2 entries.
subjects = subjects[~subjects['code'].isin(['ZZZ','SAB'])]
subjects['code'] = subjects['code'].astype('int')
worst_20 = worst_20.merge(subject_memberships, how = 'left',on = 'course_offering_uuid')
worst_20 = worst_20.merge(subjects, how = 'left',left_on= 'subject_code',right_on= 'code')
worst_20 = worst_20.drop(columns =['code','abbreviation'])
worst_20

combine1 = worst_20.merge(sections, on = 'course_offering_uuid')
combine2 = combine1.merge(teachings,how = 'left', left_on='uuid',right_on='section_uuid')
combine2 = combine2.drop(columns = ['section_uuid'])
combine3 = combine2.merge(instructors,how = 'left', left_on='instructor_id',right_on='id')
combine3_math = combine3[combine3['name_x'] == 'Mathematics']
combine3.head(3)
math_prof_score = pd.DataFrame(combine3_math.groupby(by = ['name_y']).agg({'a_count':'mean',
'ab_count':'mean',
'b_count':'mean',
'bc_count':'mean',
'c_count':'mean',
'd_count':'mean',
'f_count':'mean',
'total_count':'mean','name_y':'count'}).sort_values(['f_count',
                                                     'd_count',
                                                           'c_count','b_count'],
                                                    ascending=False))
math_prof_score = math_prof_score.rename(columns = {'name_y':'total_classes'})
math_prof_score.style.format({'a_count':"{:.2%}",
                              'ab_count':"{:.2%}",
                              'b_count':"{:.2%}",
                             'bc_count':"{:.2%}",
                             'c_count':"{:.2%}",
                             'd_count':"{:.2%}",
                             'f_count':"{:.2%}"})

schedule_merge = schedules.merge(sections[['course_offering_uuid','schedule_uuid']], how = 'left',left_on='uuid',right_on= 'schedule_uuid')
grade_merge = grade_distributions[['course_offering_uuid', 'section_number', 'a_count', 'ab_count',
       'b_count', 'bc_count', 'c_count', 'd_count', 'f_count']].copy()
test = schedule_merge.merge(grade_merge, how = 'left', on = 'course_offering_uuid')
#remove all the classes which are not scheduled 
test = test[~
((test['tues'] == False) &
(test['sun'] == False) &
(test['mon'] == False) &
(test['wed'] == False) &
(test['thurs'] == False) &
(test['fri'] == False) &
(test['sat'] == False)) ]
print(test.shape)
test_rep = test.replace({True:1,False:0})
test_rep['no_cls'] = test_rep['mon'] +test['tues']+test['sun']+test['wed']+test['thurs']+test['fri']+test['sat']
test_rep = grade_avg(test_rep)
# no significant results to say no of classes affect the grades
test_rep.groupby(by = test_rep['no_cls']).agg({'a_count':'mean',
'ab_count':'mean',
'b_count':'mean',
'bc_count':'mean',
'c_count':'mean',
'd_count':'mean',
'f_count':'mean',
'uuid':'count'})
days_affect = test_rep[test_rep['no_cls']==1].copy()
x = days_affect[['mon','tues','wed','thurs','fri','sat','sun']].idxmax(axis = 1)
days_affect['day'] =x
days_affect = days_affect[days_affect['a_count'].notnull()]
days_affect_grop = days_affect.groupby(by = days_affect['day']).agg({'a_count':'mean',
'ab_count':'mean',
'b_count':'mean',
'bc_count':'mean',
'c_count':'mean',
'd_count':'mean',
'f_count':'mean',
'uuid':'count'})
days_affect_grop
Grade_section = grade_distributions.merge(sections, how = 'left',on = 'course_offering_uuid')
grade_prof = Grade_section.merge(teachings, how = 'left',left_on = 'uuid', right_on = 'section_uuid')
strict_instructor = grade_prof[['course_offering_uuid','a_count', 'ab_count',
       'b_count', 'bc_count', 'c_count', 'd_count', 'f_count', 'instructor_id']].copy()
strict_instructor = grade_avg(strict_instructor)

strict_instructor = strict_instructor.merge(instructors,how = 'left', left_on='instructor_id',right_on= 'id')
strict_instructor.columns
strict_instructor = strict_instructor.groupby(by = strict_instructor['name']).agg({'a_count':'mean',
'ab_count':'mean',
'b_count':'mean',
'bc_count':'mean',
'c_count':'mean',
'd_count':'mean',
'f_count':'mean',
'course_offering_uuid':'count'}).sort_values(['f_count','d_count',
                                                           'c_count','b_count'],ascending=False)

strict_instructor[strict_instructor['course_offering_uuid'] >5].head(10)