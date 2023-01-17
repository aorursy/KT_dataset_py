import csv
class Customer:
    def __init__(self, _id, name, gender, birthday, email, active):
        self.id = _id
        self.name = name
        self.gender = gender
        self.birthday = birthday
        self.email = email
        self.active = active
customers = []

with open('customers.csv', 'r') as f:
    records = csv.reader(f, delimiter=';')

    next(records)

    for record in records:
        _id = record[0]
        name = record[1]
        gender = record[2]
        birthday = record[3]
        email = record[4]
        active = record[5]

        customers.append(Customer(_id, name, gender, birthday, email, active))
actives = []

for customer in customers:
    if customer.active == 'True':
        actives.append(customer)

actives.sort(key=lambda customer: customer.id)

with open('actives.csv', mode='w') as actives_file:
    actives_writer = csv.writer(actives_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    for customer in actives:
        actives_writer.writerow([customer.id, customer.name, customer.gender, customer.birthday, customer.email, customer.active])
male_inactives = []

for customer in customers:
    if customer.active != 'True' and customer.gender == 'Male':
        male_inactives.append(customer)

male_inactives.sort(key=lambda customer: customer.name)

with open('male_inactives.txt', 'w', encoding='utf-8') as output:
    for customer in male_inactives:
        output.write(customer.id + ';' + customer.name + ';' + customer.gender + ';' + customer.birthday + ';' + customer.email + ';' + customer.active)
        output.write('\n')