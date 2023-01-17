import csv
class Customer:
    def __init__(self, _id, name, gender, birthday, email, active):
        self.id = _id
        self.name = name
        self.gender = gender
        self.birthday = birthday
        self.email = email
        self.active = active
def read_csv(file):
    customers = []
    actives = []
    male_inactives = []
    
    with open(file, 'r') as f:
        reader = csv.reader(f)
        
        #pula o header
        next(reader)
        
        for line in reader:
            # está lendo como string mesmo com o csv.reader, entao, tratei assim
            line = line[0].split(';')
            
            _id = line[0]
            name = line[1]
            gender = line[2]
            birthday = line[3]
            email = line[4]
            active = line[5]
            
            # all customers
            customers.append(Customer(_id, name, gender, birthday, email, active))
            
            if active == 'True':
                # actives customers
                actives.append(Customer(_id, name, gender, birthday, email, active))
            elif gender == 'Male':
                # male inactives customers
                male_inactives.append(Customer(_id, name, gender, birthday, email, active))
                
    # sort actives by id
    actives.sort(key=lambda x: x.id)
    # sort male inactives by name
    male_inactives.sort(key=lambda x: x.name)

    with open('actives.csv', mode='w') as actives_file:
        actives_writer = csv.writer(actives_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for a in actives:
            actives_writer.writerow([a.id, a.name, a.gender, a.birthday, a.email, a.active])
            
    with open('male_inactives.txt', 'w', encoding='utf-8') as output:
        for a in actives:
            output.write(a.id + ';' + a.name + ';' + a.gender + ';' + a.birthday + ';' + a.email + ';' + a.active)
            output.write('\n')
read_csv('../input/customers/customers.csv')
