# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load in 



import numpy as np # linear algebra

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from datetime import date

import matplotlib.pyplot as plt; plt.rcdefaults()





# Input data files are available in the "../input/" directory.

# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))



# Any results you write to the current directory are saved as output.

# Import .csv file

athletes = pd.read_csv('../input/athletes.csv',low_memory=False);
# Get random countries and update them if they have different names

athletes.name = athletes.name.str.lower();

athletes.nationality = athletes.nationality.str.lower();

athletes.sex = athletes.sex.str.lower();

athletes.sport = athletes.sport.str.lower();





'''

10/17/1969

mm/dd/yyyy

'''

def get_age(dob):

    

    

    try:

        if(len(dob.strip()) == 0):

            return -1;

    

        

        born = dob.split('/');

        

        if(len(born) != 3):

            return -2;

        

        born_year = int(born[2]);

        

        # as dob returns only 2digts, the below hack is used

        if(len(born[2]) == 2 and int(born[2]) > 16):

            born_year = int('19'+str(born[2]));

        elif(len(born[2]) == 2 and int(born[2]) <= 16):

            born_year = int('20'+str(born[2]));

        

        

        

        born_month = int(born[0]);

        born_date = int(born[1]);

        

        today = date.today();

        age = today.year - born_year - ((today.month, today.day) < (born_month, born_date))

        return age;

    except:

        return -3;





    

            

            

male_gold_medals = [];

male_gold_medals_age = [];

female_gold_medals = [];

female_gold_medals_age = [];



male_silver_medals = [];

male_silver_medals_age = [];

female_silver_medals = [];

female_silver_medals_age = [];



male_bronze_medals = [];

male_bronze_medals_age = [];

female_bronze_medals = [];

female_bronze_medals_age = [];



         

def collect_medal_points():

    

    for index, row in athletes.iterrows():

        

        

        if(row['nationality'] != 'usa'):

            continue;

        

        

        has_gold = 0;

        has_silver = 0;

        has_bronze = 0;

        

        has_gold = int(athletes.iloc[index]['gold']);

        has_silver = int(athletes.iloc[index]['silver']);

        has_bronze = int(athletes.iloc[index]['bronze']);

        

        total_individual_medals = has_gold + has_silver +  has_silver;

        

        

        

        age = get_age(row['dob']);

        if(has_gold == 1):            

            if(row['sex']  == 'male'):            

                male_gold_medals.append(100);

                male_gold_medals_age.append(age);

            if(row['sex']  == 'female'):

                female_gold_medals.append(102);

                female_gold_medals_age.append(age);

        if(has_silver == 1):            

            if(row['sex']  == 'male'):            

                male_silver_medals.append(50);

                male_silver_medals_age.append(age);

            if(row['sex']  == 'female'):

                female_silver_medals.append(52);

                female_silver_medals_age.append(age);

        if(has_silver == 1):            

            if(row['sex']  == 'male'):            

                male_bronze_medals.append(15);

                male_bronze_medals_age.append(age);

            if(row['sex']  == 'female'):

                female_bronze_medals.append(17);

                female_bronze_medals_age.append(age);        

        

        '''

        total_medal_points = 0;

        

        if(has_gold == 1):            

            total_medal_points = total_medal_points + 30;

        if(has_silver == 1):

            total_medal_points = total_medal_points + 10;

        if(has_bronze == 1):

            total_medal_points = total_medal_points + 3;

            

        global male_gold_medals;

        global female_medal_points;    

            

        if(total_medal_points >  0):

            if(row['sex']  == 'male'):

                male_gold_medals = male_gold_medals + total_medal_points;

            elif(row['sex']  == 'female'):

                female_medal_points = female_medal_points + total_medal_points;   

        '''             





collect_medal_points();





def show_graph():

    

    alpha_value = 0.4;

    size_value  = 35;   

    

        

    # Plot a scatterplot of the data

    plt.scatter(male_gold_medals, male_gold_medals_age, color = 'r', s = size_value, marker='x', alpha = alpha_value, label = 'male - gold'); #male    

    plt.scatter(female_gold_medals, female_gold_medals_age, color = 'g', alpha = alpha_value, marker='x', s = size_value, label = 'female - gold'); #female

    

    plt.scatter(male_silver_medals, male_silver_medals_age, color = 'r', marker='>', s = size_value, alpha = alpha_value, label = 'male - silver'); #male

    plt.scatter(female_silver_medals, female_silver_medals_age, color = 'g', marker='>', s = size_value, alpha = alpha_value, label = 'female - silver'); #female

    

    plt.scatter(male_bronze_medals, male_bronze_medals_age, color = 'r', s = size_value, alpha = alpha_value, label = 'male - bronze'); #male

    plt.scatter(female_bronze_medals, female_bronze_medals_age, color = 'g', s = size_value, alpha = alpha_value, label = 'female - bronze'); #female

        

    plt.xlabel('Bronze, Silver, Gold')

    plt.yticks([15, 20, 22, 24, 26, 28, 30,  32, 35, 40, 45, 50, 55])

    plt.xticks([])

    plt.ylabel('age');

    

    plt.legend(loc='upper right')

        

    plt.show();

    

        

show_graph();    


