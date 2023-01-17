'''You have graduated from a computer science course and have become an amazing python programmer
and now have a great job in software engineering!
You move to Bengaluru and decide that you want to start saving to buy a house. As housing
prices are very high in Bengaluru, you realize you are going to have to save for several
years before you can afford to make the down payment(advance amount) on a house.
In Part A, we are going to determine how long it will take you to save enough money to make
the down payment given the following assumptions:

1. Call the cost of your dream home total_cost.
2. Call the portion of the cost needed for a down payment portion_down_payment. For
simplicity, assume that portion_down_payment = 0.25 (25%).
3. Call the amount that you have saved thus far current_savings. You start with a current
savings of ₹0.
4. Assume that you invest your current savings wisely, with an annual return of r (in other
words, at the end of each month, you receive an additional current_savings*r/12 funds to
put into your savings – the 12 is because r is an annual rate). Assume that your investments
earn a return of r = 0.04 (4%).
5. Assume your annual salary is annual_salary.
6. Assume you are going to dedicate a certain amount of your salary each month to saving
for the down payment. Call that portion_saved. This variable should be in decimal form (i.e.
0.1 for 10%).
7. At the end of each month, your savings will be increased by the return on your investment,
plus a percentage of your monthly salary (annual salary / 12). 

Now we will be writing a code to calculate how many months it will take you to save up enough money for
a down payment.
The following inputs are to be taken: 

1. The starting annual salary (annual_salary)
2. The portion of salary to be saved (portion_saved)
3. The cost of your dream home (total_cost)'''

def input_func():
  annual_salary = float(input("Enter your annual salary: "))
  portion_saved = float(input("Enter the percent of your salary to save, as a decimal: "))
  total_cost = float(input("Enter the cost of your dream home: "))
  month_calc(annual_salary, portion_saved, total_cost)

def month_calc(annual_salary, portion_saved, total_cost):
  portion_down_payment = 0.25 * total_cost
  current_savings = 0
  monthly_salary = annual_salary / 12
  r = 0.04
  months = 0
  while(current_savings < portion_down_payment):
    current_savings = current_savings + (r / 12) * current_savings + portion_saved * monthly_salary
    months = months + 1
  print("Number of months:", months)

input_func()
'''In the above code we unrealistically assumed that your salary didn’t change. But you are a
competent programmer now, and clearly you are going to be worth more to your company
over time! So we are going to build on your code above by factoring in a raise every six
months.

Modify your program to include the following
1. Have the user input a semi-annual salary raise semi_annual_raise (as a decimal
percentage)
2. After the 6th month, increase your salary by that percentage. Do the same after the 12th
th month, the 18th month, and so on. Write a program to calculate how many months it will
take you save up enough money for a down payment. LIke before, assume that your
investments earn a return of r = 0.04 (or 4%) and the required down payment percentage is
0.25 (or 25%). Have the user enter the following variables:
1. The starting annual salary (annual_salary)
2. The percentage of salary to be saved (portion_saved)
3. The cost of your dream home (total_cost)
4. The semiannual salary raise (semi_annual_raise)

Now we will be writing a code to calculate how many months it will take you to save up enough money for
a down payment.'''

def input_func():
  annual_salary = float(input("Enter your annual salary: "))
  portion_saved = float(input("Enter the percent of your salary to save, as a decimal: "))
  total_cost = float(input("Enter the cost of your dream home: "))
  semi_annual_raise = float(input("Enter the semiannual raise, as a decimal: "))
  month_calc(annual_salary, portion_saved, total_cost, semi_annual_raise)

def month_calc(annual_salary, portion_saved, total_cost, semi_annual_raise):
  portion_down_payment = 0.25 * total_cost
  current_savings = 0
  monthly_salary = annual_salary / 12
  r = 0.04
  months = 0
  while(current_savings < portion_down_payment):
    if(months==0 or months%6!=0):
      current_savings = current_savings + (r / 12) * current_savings + portion_saved * monthly_salary
      months = months + 1
    else:
      monthly_salary += monthly_salary*semi_annual_raise
      current_savings = current_savings + (r / 12) * current_savings + portion_saved * monthly_salary
      months = months + 1

  print("Number of months:", months)

input_func()

'''In the above program you had a chance to explore how both the percentage of your salary that you save
each month and your annual raise affect how long it takes you to save for a down payment.
This is nice, but suppose you want to set a particular goal, e.g. to be able to afford the down
payment in three years. How much should you save each month to achieve this? In this
problem, you are going to write a program to answer that question. To simplify things,
assume:

1. Your semi-annual raise is .07 (7%)
2. Your investments have an annual return of 0.04 (4%)
3. The down payment is 0.25 (25%) of the cost of the house
4. The cost of the house that you are saving for is ₹10 Lac. You are now going to try to find
the best rate of savings to achieve a down payment on a ₹10 Lac house in 36 months.
Since hitting this exactly is a challenge, we simply want your savings to be within ₹100 of
the required down payment.

We will write a program to calculate the best savings rate, as a function of your starting
salary. You should use bisection search to help you do this efficiently. You should keep track
of the number of steps it takes your bisections search to finish. You should be able to reuse
some of the code you wrote for part B in this problem. Because we are searching for a
value that is in principle a float, we are going to limit ourselves to two decimals of accuracy
(i.e., we may want to save at 7.04% or 0.0704 in decimal – but we are not going to worry
about the difference between 7.041% and 7.039%). This means we can search for an
integer between 0 and 10000 (using integer division), and then convert it to a decimal
percentage (using float division) to use when we are calculating the current_savings after 36
months. By using this range, there are only a finite number of numbers that we are searching
over, as opposed to the infinite number of decimals between 0 and 1. This range will help
prevent infinite loops. The reason we use 0 to 10000 is to account for two additional decimal
places in the range 0% to 100%. Your code should print out a decimal (e.g. 0.0704 for
7.04%). Try different inputs for your starting salary, and see how the percentage you need to
save changes to reach your desired down payment. Also keep in mind it may not be
possible to save a down payment in a year and a half for some salaries. In this case your
function should notify the user that it is not possible to save for the down payment in 36
months with a print statement.'''

def input_func():
  annual_salary = float(input("Enter the starting salary: "))
  month_calc(annual_salary)

def month_calc(annual_salary):
  semi_annual_raise = 0.07
  r = 0.04
  total_cost = 1000000
  portion_down_payment = 0.25 * total_cost
  n = 0
  current_savings = 0
  low = 0
  high = 10000
  f = (high + low)//2

  while abs(current_savings - portion_down_payment) >= 100:
    current_savings = 0
    newannual_salary = annual_salary
    g = f/10000
    for month in range(36):
      if month % 6 == 0 and month > 0:
        newannual_salary += newannual_salary*semi_annual_raise
      monthly_salary = newannual_salary/12
      current_savings += monthly_salary*g+current_savings*r/12
    if current_savings < portion_down_payment:
        low = f
    else:
        high = f
    f = (high + low)//2
    n += 1
    if n > 13:
        break
  if n > 13:
   print('It is not possible to pay the down payment in three years.')
  else:
   print('Best savings rate:', g)
   print('Steps in bisection search:', n)

input_func()