import re



def get_real_name(full_name):

    real_name = ''

    surname = full_name.split(', ')[0]

    first_name = ' '.join(full_name.split(', ')[1].split(' ')[1:])

    title = full_name.split(', ')[1].split(' ')[0]

    first_name = re.sub(r'\"[^"]*\"', '', first_name).replace('()', '')

    real_name = (first_name + ' ' + surname)

    if title == 'Mrs.' or title == 'Mme.' or title == 'Lady.' or title == 'the':

        p_start = first_name.find('(')

        p_end = first_name.find(')')

        if p_start != -1 and p_end != -1:

            real_name = first_name[(p_start+1):p_end]

            if len(real_name.split()) == 1:

                real_name = (real_name + ' ' + surname)

    real_name = ' '.join(real_name.split())

    return real_name
import requests

from requests.utils import quote



def get_nat_from_name_prism(name):

    request_url = 'http://www.name-prism.com/api/json/' + quote(name)

    r = requests.get(request_url)

    try:

        data = r.json()

        # data = {"European,SouthSlavs": 2.3147843795616316e-06, "Muslim,Pakistanis,Bangladesh": 1.0081305531161525e-05, "European,Italian,Italy": 3.856736678433249e-05, "European,Baltics": 7.713378348172508e-07, ...}

    except:

        print('ERROR')

        print(r)

        return None

    max_match = 0.0

    nat = ''

    for k,v in data.items():

        if data[k] > max_match:

            max_match = data[k]

            nat = k

    return nat
import csv



file_name = 'pass_nationalities.csv'



def write_nationality_data(nat_data):

    with open(file_name, 'w') as csv_out:

        writer = csv.writer(csv_out)

        writer.writerow(['PassengerId','Nationality'])

        for k,v in nat_data.items():

            writer.writerow([k,v])

    

         

# Useful for restarting where we left off in case of an exception

def read_nationality_data():

    res = {}    

    try:

        with open(file_name) as f1:

            reader = csv.DictReader(f1)

            for row in reader:

                res[row['PassengerId']] = row['Nationality']

    except:

        print('no file to start')

    return res





def extract_nat_data():

    known_nat_data = read_nationality_data() 

    files = ['train.csv', 'test.csv']

    for fname in files:

        with open(fname) as f1:

            reader = csv.DictReader(f1)

            for row in reader:

                pid = row['PassengerId']

                real_name = get_real_name(row['Name'])

                if not pid in known_nat_data:

                    nat_data = get_nat_from_name_prism(real_name)

                    if nat_data != None:

                        known_nat_data[pid] = nat_data

                write_nationality_data(known_nat_data) # write after each request in case program crashes