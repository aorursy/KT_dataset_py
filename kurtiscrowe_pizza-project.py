import sys

# Pizza Planet order
print('Welcome to Pizza Planet! Home of the BEST prison pizza!!')
# list of prices
price_size = {'Small': 5.00, 'Medium': 10.00, 'Large': 15.00}
price_topping = {'Pepperoni': 1.25, 'Sausage': 2.00, 'Mystery Meat': .50}
price_cheese = {'Ricotta': .25, 'Mozzarella': .15, 'Fontina': .12}
price_sauce = {'Red Sauce': .5, 'White Sauce': .10, 'Garlic Sauce': .12}

ordering = True
total = 0

while ordering:
    print('Please select your pizza Size!')
    Size = input('Small, Medium, or Large:').title()
    if Size == 'Small':
        total = total + price_size['Small']

        print('You have selected size: Small')
    if Size == 'Medium':
        total = total + price_size['Medium']

        print('You have selected size: Medium')
    if Size == 'Large':
        total = total + price_size['Large']

        print('You have selected size: Large')

    print('Please select your topping')
    topping = input('Pepperoni, Sausage, or Mystery Meat?').title()
    if topping == 'Pepperoni':
        total = total + price_topping['Pepperoni']

        print('You have selected Pepperoni as your topping!')
    if topping == 'Sausage':
        total = total + price_topping['Sausage']

        print('You have selected Sausage as your topping!')
    if topping == 'Mystery Meat':
        total = total + price_topping['Mystery Meat']

        print('You have selected Mystery Meat as your topping!')

#CHEESE ORDER
    print('Please select your cheese: ')
    cheese = input('Ricotta, Mozzarella, Fontina: ').title()
    if cheese == 'Ricotta':
        total = total + price_cheese['Ricotta']

        print('You have selected Ricotta as your Cheese!')
    if cheese == 'Mozzarella':
        total = total + price_cheese['Mozzarella']

        print('You have selected Mozzarella as your Cheese!')
    if cheese == 'Fontina':
        total = total + price_cheese['Fontina']

        print('You have selected Fontina as your Cheese!')

#Sauce
    print('Please select your type of Sauce: ')
    sauce = input('Red Sauce, White Sauce, Garlic Sauce: ').title()
    if sauce == 'Red Sauce':
        total = total + price_sauce['Red Sauce']

        print('You have selected Red Sauce as your Sauce!')
    if sauce == 'White Sauce':
        total = total + price_sauce['White Sauce']

        print('You have selected White Sauce as your Sauce!')
    if sauce == 'Garlic Sauce':
        total = total + price_sauce['Garlic Sauce']

        print('You have selected Garlic Sauce as your Sauce!')

#Order Complete
    print("You have Completed your order, YAY!")
    print('Your total is:')
    print(total *.935)

    if total > 0:
        sys.exit('Come again next time!')