class IndividualCustomer:
    def __init__(self, name, surname, phone):
        """
        Constructor
        :param name: Name of customer
        :param surname: Surname of customer
        :param phone: Mobile phone number of customer
        """
        self.name = name
        self.surname = surname
        self.phone = phone

    def __str__(self):
        return "Name: {}, Surname: {}, Phone: {}".format(self.name, self.surname, self.phone)

    def show_phone(self):
        print("Phone: {}".format(self.phone))

class Lead(IndividualCustomer):
    def show_phone(self):
        super().show_phone()
        print("Remember that this is a lead customer")

    def __del__(self):
        print("Lead customer object is destroyed")
class Contact(IndividualCustomer):
    pass
ind1 = IndividualCustomer("Alex", "Sanchez", "532112233")

print("""
*************************
Individual Customer:
*************************""")
print(ind1)
ind1.show_phone()
cnt1 = Contact("Martin", "Fowler", "532443322")

print("""
*************************
Contact:
*************************""")
print(cnt1)
cnt1.show_phone()
lead = Lead("Rob", "McQuinn", "5357778899")

print("""
*************************
Lead:
*************************""")
print(lead)
lead.show_phone()
lead.__del__()