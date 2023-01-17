'''

# SETUP. You don't need to worry for now about what this code does or how it works.
from learntools.core import binder; binder.bind(globals())
from learntools.python.ex2 import *
print('Setup complete.')

'''

#Program to find the Day when a Date is given

#          Author: Praveen Reddy. M          #
#          UserName: pkrmarthala          #

# For the complete Procedure, formula and explanation, please visit the website
#                       praveenzneuro.blogspot.com

def DefineDay(k, m, y):
    '''
    The function takes in the values of the Date, Month, and Year;
    Processes them and return the Output.

    Eg.:
    Enter the Year (in the yyyy format): 1999
    Enter the Month (in the mm format): 11
    Enter the Date (in the dd format): 26
    The date 26 / 11 / 1999 falls on  Friday
    '''
    
    #----- Logic to find the Total F value using the Formula -----#
    if (m>2 and m<12):
        m=m-2;
        #print(m); #To check the 'm' value we are about to submit in the formula
    else:
        m+=10;
        y -= 1;
        #print('Year: ',y); #To check the 'Year' value we are about to submit in the formula
        #print('month: ',m); #To check the 'm' value we are about to submit in the formula
    c = int(y//100);
    d = int(y%100);
       
    f= int(k+((13*m-1)//5)+d+(d//4)+(c//4)-(2*c));
    #print ('The Final Value is: ',f); #To know the final 'F' value

    #-----Logic to find the Remainder and decide the Day-----#
    rem = int(f%7);
    if (rem==0):
        dayy = 'Sunday';
    elif(rem==1):
        dayy = 'Monday';
    elif(rem==2):
        dayy = 'Tuesday';
    elif(rem==3):
        dayy = 'Wednesday';
    elif(rem==4):
        dayy = 'Thursday';
    elif(rem==5):
        dayy = 'Friday';
    else:
        dayy = 'Saturday';

    return dayy;


#----------------  The End of the Function ----------------#





while True:
    #----- Getting Input (Year) from the User/Tester -----#
    year = int(input('Enter the Year (in the yyyy format): '));


    #----- Getting the Month value from the User -----#
    month = int(input('Enter the Month (in the mm format): '));

    
    #----- Logic to check whether the entered Month is valid or not -----#
    if(month > 12 or month < 1):
        print("Please enter a valid month value between 1 and 12\n\n");
        continue;
    

    #----- Getting Date from the User/Tester -----#
    date = int(input('Enter the Date (in the dd format): '));


    #----- Logic to check whether the entered Date is correct or not -----#
    if(date > 31 or date < 1):
        print("The date",date,"does not exist for any month!\nPlease try again by entering a valid date!\n\n");
        continue;


    #----- Date check for the month of February -----#    
    elif (month == 2):
        if (((year % 4 == 0 or year % 400 == 0) and date > 29) or date > 28): #one-line code to check date validity for the Leap and Non-Leap Years
            print("The date",date,"does not exist for the month!\nPlease try again by entering a valid date!\n\n");
            continue;
        #elif (date > 28): #Uncomment this line and remove the last 'or' part of the above statement to use this line separately
            #print("The date",date,"does not exist for the month!\nPlease try again by entering a valid date!\n\n");
        else:
            #print("In the Else Part!"); #To check for the Program Ececution
            day = DefineDay(date, month, year); #Function Call
            print('The date',date,'/',month,'/',year,'falls on ',day); #Printing the Result

            
    #----- Date check for the Odd Months -----#
    elif (month == 1 or month == 3 or month == 5 or month == 7 or month == 8 or month == 10 or month == 12):
        if (date > 31):
            print("The date",date,"does not exist for the month!\nPlease try again by entering a valid date\n\n!");
            continue;
        else:
            #print("In the Else Part!"); #To check for the Program Ececution
            day = DefineDay(date, month, year); #Function Call
            print('The date',date,'/',month,'/',year,'falls on ',day); #Printing the Result


    #----- Date check for the even Months -----#
    elif (date > 30):
        print("The date",date,"does not exist for this month!\nPlease try again by entering a valid date!\n\n");
        continue;


    else:
        #print("In the Else Part!"); #To check for the Program Ececution
        day = DefineDay(date, month, year); #Function Call
        print('The date',date,'/',month,'/',year,'falls on ',day); #Printing the Result
        

    #----- Prompt the User if he wants to test another Date -----# 
    ch = str(input("\n\nDo you want to Continue? [Y/N]: "));
    #print(ch);
    if(ch == 'N'):
        break;
    else:
        print("\n");
        continue;
print("Thank You!");