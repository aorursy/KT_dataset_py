resident = input("Are you resident for tax purposes (Y/N)?:")
salary = input("Enter your taxable inome (in dollars): ")
taxable_income = int(salary)
if((resident == "Y") or (resident == "y")):
    if(taxable_income>= 180001):
        tax = 54232 + .45*(taxable_income - 180000)
        print("Your tax is :${0}".format(str(tax)))
    elif(taxable_income>= 87001):
        tax = 19822 + .37*(taxable_income - 87000)
        print("Your tax is :${0}".format(str(tax)))
    elif(taxable_income>= 37001):
        tax = 572 + .325*(taxable_income - 37000)
        print("Your tax is :${0}".format(str(tax)))
    elif(taxable_income>= 18201):
        tax = 19822 + .19*(taxable_income - 18200)
        print("Your tax is :${0}".format(str(tax)))
    else:
        print("Your tax is : Nil")
else:
    if(taxable_income>= 180001):
        tax = 62685 + .45*(taxable_income - 180000)
        print("Your tax is :${0}".format(str(tax)))
    elif(taxable_income>= 87001):
        tax = 28275 + .37*(taxable_income - 87000)
        print("Your tax is :${0}".format(str(tax)))
    else:
        tax = .325*taxable_income
        print("Your tax is :${0}".format(str(tax)))    