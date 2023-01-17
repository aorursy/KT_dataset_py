# This Python 3 environment comes with many helpful analytics libraries installed

# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python

# For example, here's several helpful packages to load



import csv



# Input data files are available in the read-only "../input/" directory

# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



import os

for dirname, _, filenames in os.walk('/kaggle/input'):

    for filename in filenames:

        print(os.path.join(dirname, filename))



# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 

# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
class Customer:

    def __init__(self, _id, name, gender, birthday, email, active):

        self.id = _id

        self.name = name

        self.gender = gender

        self.birthday = birthday

        self.email = email

        self.active = active
def get_customers(file):

    customers = []

    

    with open(file, 'r') as f:

        reader = csv.reader(f, delimiter=';')       

        

        for i, line in enumerate(reader):

            if i == 0:

                continue

            customers.append(Customer(line[0], line[1], line[2], line[3], line[4], line[5]))

            

    return customers
def save_customers_csv(customers, filename):

    with open(filename, mode='w') as file:

        actives_writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for a in customers:

            actives_writer.writerow([a.id, a.name, a.gender, a.birthday, a.email, a.active])          
def save_customers_txt(customers, filename):

    with open(filename, 'w', encoding='utf-8') as output:

        for c in customers:

            output.write(';'.join((c.id, c.name, c.gender, c.birthday,c.email, c.active,'\n')))
customers = get_customers('/kaggle/input/customers.csv')



actives = list(c for c in customers if c.active == "True") 

actives.sort(key=lambda x: x.id)



male_inactives = list(c for c in customers if c.active == "False" and c.gender == "Male")

male_inactives.sort(key=lambda x: x.name)



save_customers_csv(actives, 'actives.csv')

save_customers_txt(male_inactives, 'male_inactives.txt')