import pandas as panda
import numpy as np

## lets create a very simple data frame

animals = ['horse','cat','dog','tiger','whale','dog','tiger','horse','cat','rat','cat']
weight = [245,12,15,200,780,17,176,234,12,9,15]
age = [14,15,4,17,34,2,21,21,2,5,15]
a = panda.DataFrame(
    {
        'animal': animals,
        'weight': weight,
        'age' : age,
        # 'color' : ['']

    }
)

a
group_by_animal = a.groupby(by = ['animal'])

## groupby returns pandas.core.groupby.DataFrameGroupBy or SeriesGroupBy
## when resultant is multi columns - DataFrameGroupBy
## when resultant is single column - SeriesGroupBy
## try this and see a = panda.DataFrame({'id':np.random.randint(1,10,(6))})
##a.groupby(by=['id'])

print(type(group_by_animal)) ## 


## what if i want to know the group names
print('\n The group names are :',group_by_animal.groups.keys())


##what if i want to know how the grouping has been done
## essentially which are the groups..which indices form those groups
print('\n My different groups and the indices they belong to :',group_by_animal.groups)



## how do i view the groupby.. one very simple method is list
print('\n i would like to see the group : \n' , list(group_by_animal))


## view by iterating

for name, group in group_by_animal:
    print('\n Group name: ', name , ' \n group is :', group)
group_by_animal_and_age = a.groupby(['animal','age'])

for name, group in group_by_animal_and_age:
    print('\n name is :', name)
    print('\n group is : \n', group)
i_know_group_key = 'cat'
i_know_another_group_key = ('whale', 34)
print(group_by_animal.get_group(i_know_group_key),'\n\n', group_by_animal_and_age.get_group(i_know_another_group_key))


## group_by_animal['age'] is simple syntactic sugar.

group_by_animal['age'].max()

# or you may you aggregate
group_by_animal['age'].agg(np.max)
## example using aggregate
group_by_animal['age'].aggregate(np.max)
## pass in a list of aggregation function in the add parameter
group_by_animal['age'].agg([np.max, np.mean])
## you will notice in the above example that our group keys becomes our index.
## If you would like to change that, simply use reset_index
group_by_animal['age']\
    .agg([np.max, np.mean])\
        .reset_index()

# i do not like the column names of the output .
# i would like to give it my own column names

group_by_animal['age'].agg({'i_am_the_max_age':np.max,'i_am_the_mean_age':np.mean})
# another way to rename the columns using method chaining

group_by_animal['age']\
    .agg([np.max, np.mean])\
        .rename(columns = {'amax':'i am the max age', 'mean': 'i am the mean age'})\
            .reset_index()
my_own_custom = lambda x : np.mean(x)-10
group_by_animal['age']\
        .agg({'max':np.max,'mean':np.mean,'mean_minus_10':my_own_custom})\
            .reset_index()
group_by_animal_and_age['weight'].agg(np.max)
group_by_animal_and_age['weight'].agg(np.max).reset_index()
mean_age = group_by_animal['age']\
        .agg({'mean_age':np.mean})\
                .reset_index()
print(mean_age.shape, a.shape) ## you will see very clearly how the row counts differ

print(mean_age)
## we merge the mean back with the original data set
b = a.merge(mean_age, how='inner', on='animal')
b['diff_in_age_with_mean'] = b.age -b.mean_age
b
a["diff_in_age_from_mean"] = group_by_animal["age"].transform(lambda x: x -x.mean())
a
a.index = a.animal
a
def is_animal_ending_with_at(animal_name):
    if animal_name.endswith('at'):
        return 'animals_ending_with_at'
    else:
        return 'other animals'

list(a.groupby(by = is_animal_ending_with_at))
    