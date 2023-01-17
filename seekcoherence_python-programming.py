# Desc: This program gathers company and order information to generate a receipt for the buyer.





print('Hello welcome to the TechnoFiber order form')



# Gathers customer order information

cust_company = input('Please enter your company name \n')



cust_ftcable = input('How much feet of Fiber optic cable would you like? \n')



# Ensures only numbers are entered not strings or abbreviations IE.. FT or ", '

try:

    cust_ftcable = float(cust_ftcable)

except:

    print('Please enter numbers only IE 10 or 10.23 -Abbreviations or words are not allowed ')

    print('Order form will close please try again. press ENTER to continue')

    input()

    exit()



# Calculates price given the total amount of feet customer inputs

if cust_ftcable <= 100:

    price = cust_ftcable * 0.87

elif cust_ftcable <= 250:

    price = cust_ftcable * 0.80

elif cust_ftcable <= 500:

    price = cust_ftcable * 0.70

else:

    price = cust_ftcable * 0.50



# Determines discount given depending on qty feet ordered

if cust_ftcable <= 100:

    dis_reg = 0.87

elif cust_ftcable <= 250:

    dis_reg = 0.80

elif cust_ftcable <= 500:

    dis_reg = 0.70

else:

    dis_reg = 0.50



total_cost = price * dis_reg



# Creates a receipt for customer

print('Your order has been submitted \n')

print('~~Order receipt for ' + cust_company +'~~')

print('Length of feet to install: ' + str(cust_ftcable))

print('Calculated cost: ' + str(cust_ftcable) + '/Feet' + ' X ' + '$' + format(dis_reg,',.2f') + '/price per foot')

print('Total Cost: ' + '$'+ format(total_cost,',.2f'))

print("Thank you for your business!")



# Prevents windows terminal from closing automatically

input()