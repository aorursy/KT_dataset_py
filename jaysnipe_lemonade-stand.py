# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import IPython.display as disp
import random
from matplotlib import pyplot as plt

# Easy Mode
sugar_cost = 2.0
lemon_cost = 0.25
advertising_cost = 100
# Hard Mode
sugar_cost = 4.0
lemon_cost = 0.50
advertising_cost = 200

lemonade_pitcher_cost = lemon_cost * 4 + sugar_cost / 2

loan_officers = ['Fredo', 'Bubba', 'Al', 'Bugsy', 'Mack', 'Dexter', 'The Grish', 'The Greek']

sugar_amount = 10.0
lemon_amount = 20
money = 50.0
loan = 0
money_history = [50]
aj_money_history = [50]
day_history = [0]
weather_report_quality = 1
day_cnt = 1
time_of_day = "AM"
advertising_factor = 1
advertising_days = 0
competition_vacation_days = 0


def set_price_curve():  
    #  RETURNS A PRICE CURVE
    #  May be edited to pull from database or text file
    #
    #     temperature determines the max number of glasses you could sell
    # above 50 you can sell 25 cups for a quarter
    # above 60 you can sell 50 cups for a quarter and 25 for $1.25
    # ...
    # above 110 you can sell 300 cups for a quarter, 300 for $1.25, 250 for $2.25, 200 for $3.25, 100 for $4.25, and 50 for $5
    # CAN I USE MACHINE LEARNING TO PREDICT THE AMOUNT SOLD?
    curve = {
        "110:1.25": 300,
        "110:2.25": 250,
        "110:3.25": 200,
        "110:4.25": 100,
        "110:5.00": 50,
        "110:6.00": 0,
        "100:0.25": 300,
        "100:1.25": 250,
        "100:2.25": 200,
        "100:3.25": 100,
        "100:4.25": 50,
        "100:5.00": 25,
        "100:6.00": 0,
        "90:0.25": 250,
        "90:1.25": 150,
        "90:2.25": 100,
        "90:3.25": 50,
        "90:4.25": 25,
        "90:5.00": 0,
        "80:0.25": 200,
        "80:1.25": 100,
        "80:2.25": 50,
        "80:3.25": 25,
        "80:4.25": 0,
        "70:0.25": 75,
        "70:1.25": 50,
        "70:2.25": 25,
        "70:3.25": 0,
        "60:0.25": 50,
        "60:1.25": 25,
        "60:2.25": 0,
        "50:0.25": 25,
        "50:1.25": 0        
    }
    return curve

def show_achievements():
    print("You don't have any achievements! Why don't you make something of your life first?")
    boot(False)

def buy_item(num, cost):
    return (num * cost) <= money

def buy_sugar():
    print("How many pounds of sugar would you like to buy at ${:.2f} per pound?".format(sugar_cost))
    
def buy_lemons():
    print("How many lemons would you like to buy at ${:.2f} per lemon?".format(lemon_cost))

def show_status():
    print("You have ${:.2f}, {:.1f} pounds of sugar, and {} lemons.".format(money,sugar_amount, lemon_amount))

def make_pitcher(pitcher_cnt):
    made_pitchers = 0
    global sugar_amount, lemon_amount
    while sugar_amount >= 0.5 and lemon_amount >= 4 and made_pitchers < pitcher_cnt:
        sugar_amount -= 0.5
        lemon_amount -= 4
        made_pitchers += 1
    if(made_pitchers < pitcher_cnt):
        print("You ran out of ingredients, but it's too late to go to the store now.")
    return made_pitchers

def buy():
    global sugar_amount
    global money
    global lemon_amount
    show_status()
    buy_sugar()
    amt = 0
    
    try:
        amt = float(input())
    except ValueError:
        print("Try putting in numbers next time")
    else:    
        while not buy_item(amt, sugar_cost):
            print("That would leave your account overdrawn by ${:.2f}".format(money - sugar_cost*amt))
            buy_sugar()
            amt = float(input())
    sugar_amount += amt
    money = money - amt * sugar_cost
    show_status()
    buy_lemons()
    try:
        amt = int(input())
    except ValueError:
        print("Try putting in numbers next time")
    else:
        while not buy_item(amt, lemon_cost):
            print("You don't have enough money!")
            buy_lemons()
            amt = int(input())
    lemon_amount += amt
    money = money - amt * lemon_cost
    show_status()
    boot(False)

def weather_report():
#     Implement ability to buy higher quality weather reports
#     Let's get a file of funny responses and select from them at random. You could even have a hard mode where they don't say the temperature
    global predicted_temp
    predicted_temp = random.randint(50,110)
    if(predicted_temp < 60):
        print("Sounds like a cold front is blowing in tomorrow, they say the high is only {}°".format(predicted_temp))
    elif(predicted_temp < 70):
        print("Meh, only a high of {}°".format(predicted_temp))
    elif(predicted_temp < 80):
        print("Sales could be heating up once it hits {}°".format(predicted_temp))
    elif(predicted_temp < 90):
        print("Time to make some money with a predicted high of {}°".format(predicted_temp))
    elif(predicted_temp < 100):
        print("Sales could be heating up once it hits {}°".format(predicted_temp))
    else:
        print("Gonna be a scorcher if it hits {}°".format(predicted_temp))
    boot(False)

def weather_actual():
    variability = 20 / weather_report_quality
    return random.randint(predicted_temp - variability,predicted_temp + variability)

def sell_lemonade(next_day = False):
#   We could have a file of random events that could happen that impact the profit of the day
    global money
    global time_of_day
    global day_cnt
    global advertising_days
    global competition_vacation_days
    time_of_day = "PM"
    if(not next_day):
        print("How many pitchers (5 glasses) do you want to make?")
        pitchers = int(input())
        pitchers_made = make_pitcher(pitchers) 
        print("You made {} glasses of lemonade".format(pitchers_made * 5))
        print("How much do you want to charge per glass?")
        price = float(input())
    else:
        pitchers_made = 0
        price = 6.00
    money = money + price_logic(price, pitchers_made)
    day_cnt += 1
    if(advertising_days > 0):
        advertising_days -= 1
    if(advertising_days == 0):  # once advertising days is corrected, remove advertising bonus if appropriate
        advertising_factor = 1
    if(competition_vacation_days > 0):
        competition_vacation_days -= 1
    print_header()
    boot(False)    

def get_max_cups(price, degrees, market_factor):
    # get the number of cups sold based on the temperature and price and the market
    ret_cups = 0
    for priceIndex, marketMax in price_curve.items():
        heat,prc = priceIndex.split(':')
        if ( (price <= float(prc)) and (degrees >= float(heat)) ):
            ret_cups = marketMax * market_factor * advertising_factor # Most cups that can be sold at that point * your market_factor * advertising factor
            break
    return ret_cups
    
def get_evil_results(degrees, comp_price, competition_factor):
    # get the amount of money your competitor (Evil John) made
    cups = get_max_cups(comp_price, degrees, competition_factor)
    return cups  

def get_evil_price(degrees):
    # Evil John has perfected his JIT ordering and lemonade-making, but he's a bit of a git when it comes to setting pricing
    if(degrees > 100):
        John_max_price = 4.25
    elif (degrees > 90):
        John_max_price = 3.25
    else:
        John_max_price = 3.25
    price_percentage = random.randint(1,99) / 100
    return 0.25 + John_max_price * price_percentage

def comp_factor(your_price, his_price, pitcher_cnt):
    # Whoever has the lower price gets a 50% boost to sales, unless the other guy isn't selling at all - then you get a 100% boost
    if(competition_vacation_days == 0):
        if(pitcher_cnt == 0):
            return 0.0, 2.0
        if(your_price > his_price):
            return 0.5, 1.5
        elif(his_price > your_price):
            return 1.5, 0.5
        else:
            return 1.0, 1.0
    else:
        return 2.0, 0

def price_logic(price, pitchers):
    real_temp = weather_actual()
    evil_price = get_evil_price(real_temp)
    # get results of price war
    your_factor, evil_factor = comp_factor(price, evil_price, pitchers)
    
    sold_cups = 0
    max_cups = get_max_cups(price, real_temp, your_factor)
    made_cups = pitchers * 5
    if(made_cups > max_cups): #If you made more cups than you could sell, you were limited by the price. Otherwise you were limited by cups made
        sold_cups = max_cups
    else:
        sold_cups = made_cups
    disp.clear_output()
    
    gross_sales = sold_cups * price
    total_cost = lemonade_pitcher_cost * pitchers
    net_profit = (gross_sales - total_cost) - loan * 0.2
    if(competition_vacation_days == 0):
        evil_cups = get_evil_results(real_temp, evil_price, evil_factor)
        evil_cost = (lemonade_pitcher_cost / 5) * evil_cups
        evil_profit = evil_cups * evil_price - evil_cost
    else:
        evil_cups, evil_cost, evil_profit = 0
    
    # write these to history
    global money_history
    global aj_money_history
    global day_history
    old_money = money_history[-1]
    money_history.append(old_money+net_profit)
    day_history.append(day_cnt)
    old_money = aj_money_history[-1]
    aj_money_history.append(old_money + evil_profit)
    if(pitchers > 0): #You chose to operate
        print("Time to close up shop for the day. The high for the day was {}°.".format(real_temp))
        print("You sold {} cups of lemonade at ${:.2f}, making ${:.2f}".format(sold_cups, price, gross_sales))   
        print("Your costs were: \nLemonade: {} pitchers @ ${:.2f} per pitcher = ${:.2f}".format(pitchers, lemonade_pitcher_cost, total_cost))
    else: 
        print("Didn't feel like working today, huh?")
        print("Your lemonade making costs were $0.00")
    if(loan > 0):
        print("You also had to pay {} ${:.2f}".format(get_loan_officer(), loan*0.2))        
    print("Your net profit was ${:.2f}".format(net_profit))
    print("---------------------------")
    print("Angry Johnny sold {} cups at ${:.2f}".format(evil_cups, evil_price))
    print("Angry Johnny made a net profit of ${:.2f}".format(evil_profit))
    display_progress()
    return sold_cups * price


def display_progress():
    x = day_history
    y = money_history
    x2 = day_history
    y2 = aj_money_history
    plt.plot(x,y,'g', label='Our Hero')
    plt.plot(x2,y2, 'r', label='Angry Johnny')
    plt.title('Your Progress')
    plt.legend()
    plt.show()
    
                
def FAQ():
    print("Lemonade Recipe - (per pitcher) {0.5 pounds of sugar, 4 lemons}")
    boot(False)
    
def tell_story():
    print("----------")
    print("You are a small business owner struggling in the wake of the government-inflicted Corona virus economic meltdown.")
    print("Since you can't run your Irish / Street Food gastropub (Tacos D'Terre), you have decided to open a lemonade stand.")
    print("Your goal is to make enough money to get a liquor license for your stand. ($1,000)\n")
    print("Unfortunately, your arch enemy Angry Johnny has opened up shop as well. ")
    print("You must also make more money than him this over the next 20 days or he will use his profits to bribe a local official to shut you down.")
    print("Instructions: ")
    print("Type a word from the menu or its associated number to perform that action. ")
    print("You will find some interesting options in the Financials menu.")
    print("----------")
    # Maybe this can set up a game dynamic of economic debuffs 
                
def clear():
    boot(True)
    
def next_day():
    global time_of_day
    global day_cnt 
    time_of_day = "AM"
    day_cnt += 1
    sell_lemonade(True)

def advertise():
    print("Advertising costs ${}, but doubles your sales for the next three days\nSpend the money(Y/N)?".format(advertising_cost))
    option = input()
    if(option == "Y"):
        advertising_factor = 2
        advertising_days = 3
    show_finance(False)
        
def get_loan():
    global money
    global loan
    print("Loans cost 20% of the loan principal daily until returned. ")
    print("How much money would you like to borrow?")
    try:
        amt = float(input())
    except ValueError:
        print("Try putting in numbers next time")
    else:
        money += amt
        loan += amt
        get_loan_text()
        
def get_loan_officer():
    num = len(loan_officers) - 1
    return(loan_officers[random.randint(0,num)])

def get_loan_text():
    officer = get_loan_officer()
    print("Thanks for your business!")
    print("Our daily collection plan is included with your package!")
    print("{} will be by to collect ${:.2f} tonight".format(officer, loan * 0.2))
    txt_num = random.randint(1,3)
    if(txt_num == 1):
        print("Please don't make {0} angry. You won't like {0} when he's angry.".format(officer))
    elif(txt_num == 2):
        print("Don't let things get bumpy with {}. He likes a bit of the rumpy-pumpy.".format(officer))
    else:
        print("Princess Bride quote")
    print_header()
    boot(False)
    
def pay_loan():
    global money
    global loan
    if(loan > money):
        print("I don't think you can afford to do that")
    else:
        print("Thank you, come again!")
        loan = 0
        money -= loan
    boot(False)
        
def special():
    global money
    global competition_vacation_days
    print("Would you like us to send one of our 'Loan' officers over to visit the competition?")
    print("For a hundred bones, {} here might be able to convince the other lemonade stand to take a couple day vacation.".format(get_loan_officer()))
    print("Y/N")
    option = input().lower()
    if(option == "Y"):
        money -= 100
        v = random.randint(1,3)
        competition_vacation_days = v
        print("What do you know? They decided to take {} day{} off!".format(competition_vacation_days, 's' if v > 1 else ''))
    boot(False)

def upgrade():
    global weather_report_quality
    print("Would you like us to upgrade the accuracy of your weather reports?")
    print("For a mere $500, we promise to be only half as wrong!")
    print("Y/N?")
    option = input().lower()
    if(option == "Y"):
        money -= 500
        weather_report_quality = 2
    boot(True)
          
def finance_help():
    print("Loans:")
    print("Loans cost 20% of the loan principal daily until returned. ")
    print("Use the 'Pay Loan' option to return your loan")
    print("Advertise:")
    print("Advertising costs ${}, but doubles your sales for the next three days".format(advertising_cost))
    print("Special:")
    print("Getting tired of the competition? We might be able to help you with that.")
    print("Weather:")
    print("Upgrade the accuracy of your weather reports ($500)")
    
def print_header():
    print("Welcome to the Lemonade Stand Day {} --- Money ${:,.2f} Lemons {} Sugar {}  ---".format(day_cnt, money, lemon_amount, sugar_amount))
    print("**-----------------------------------------------------------------------**")
    print("Menu || 1:Buy | 2:Sell | 3:Financials | 4:Weather | 5:Next Day | 6:Achievements | 7:Clear | 8:Help | 9:Quit")
    
def show_finance(clear_screen = False):
    if(clear_screen):
        disp.clear_output()
    print("Welcome to Finance Day {} --- Money ${:,.2f} Lemons {} Sugar {}  ---".format(day_cnt, money, lemon_amount, sugar_amount))
    print("**-----------------------------------------------------------------------**")
    print("Finance Menu || 1:Loan | 2:Pay Loan | 3:Advertise | 4:Special | 5:Upgrade | 6:Help | 7:Return to Stand")
    option = input().lower()
    if(option == "loan" or option == "1"):
        get_loan()
    elif(option.startswith("pay") or option == "2"):
        pay_loan()
    elif(option == "advertise" or option == "3"):
        advertise()
    elif(option == "special" or option == "4"):
        special()
    elif(option == "upgrade" or option == "5"):
        upgrade()    
    elif(option == "help" or option == "6"):
        finance_help()
    elif(option == "return" or option == "7"):
        next_day()
    else:
        print("Sorry, I didn't get that")
        show_finance(False)
        
def boot(clear_screen = True):
    if(clear_screen):
        disp.clear_output()
        print_header()
        if(day_cnt == 1):
            tell_story()
    option = input().lower()
    if(money < 2 and (lemon_amount < 4 or sugar_amount < 0.5) and time_of_day == "PM"):
        print("Hey, even Trump went bankrupt. Why don't you go to Financials to talk to our cut-rate loan officers?")
    if(option == "buy" or option == "1"):
        buy()
    elif(option  == "sell" or option == "2"):
        sell_lemonade()
    elif(option == "financials" or option == "3"):
        show_finance(False)
    elif(option == "weather" or option == "4"):
        weather_report()
    elif(option.startswith("next") or option == "5"):
        next_day()
    elif(option == "achievements" or option == "6"):
        show_achievements()
    elif(option == "clear" or option == "7"):
        clear()
    elif(option == "help" or option == "8"):
        FAQ()
    else:
        print("Let's " + option)

        
price_curve = set_price_curve()
boot()