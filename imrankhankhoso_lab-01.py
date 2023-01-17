#definig "names" list that contain names
names=['asad','riaz','qamar']

#definig "names" list that contain names
heights=[5.8,5.6,5.5]

#use sort() method to convert the list of names in ascending order
names.sort()
#finding maximum height with their names using index
x1=max(heights)
x2=heights.index(x1)
#finding height height with their names using index
y1=min(heights)
y2=heights.index(y1)

#first finding length of names list and then adding all values in height list to find average height
ave=len(names)/sum(heights)
#printing names to show in ascending order, priniting total number of entries in list, finding maximum height and minimum height with names and prinitng average height
names,len(names),names[x2],x1,names[y2],y1,sum(heights),ave

#defining main method
def main():
    #calling BMI method in main method
    BMI()
#initializing paramater to 0 initially
height=0.0
weight=0.0
bmi=0.0
    
    #defining function BMI and it's parameter
def BMI():
    #taking inputs of type float for weight and height    
        weight = float(input('enter your weight in kg '))
        height = float(input('enter your height in meters '))
    #defining formula for finding BMI
        bmi= weight/(height*height)
       
    #if statement, if value of BMI is less than 18.5 it should print following
        if bmi < 18.5:
            print('your BMI is', bmi, 'you are underweight')
    #else if statement, if value of BMI is greater than and equal to 18.5 and less than 25 then it should print following
        elif 18.5 <= bmi < 25:
                print('your BMI is', bmi,'you are normal')
    #else if statement, if value of BMI is greater than and equal to 25 and less than 30 then it should print following
        elif 25 <= bmi < 30:
                print('your BMI is', bmi,'you are overweight')
    #else if statement, if value of BMI is greater than and equal to 30 then it should print following
        elif bmi >= 30:
                print('your BMI is', bmi,'you are obese')
    #calling function BMI
BMI()

#defining main method
def main():
#calling temp() method to main mathod
    temp()
#initially setting variables to zero
celsius=0.0
kelvin=0.0
fahrenheit=0.0

#defining function temp and its parameters celsius,Kelvin and fahrenheit
def temp():
#taking temperature in celsius of type float as input
    celsius = float(input('enter temperature in celsius '))
#defining formula for converting celsius into kelvin
    kelvin=celsius+273
#defining formula for converting celsius into fahrenheit
    fahrenheit=(celsius * 9/5) + 32
#prinitng temperature in kelvin and fahrenheit on output
    print('the temperature in kelvin is ',kelvin)
    print('the temperature in fahrenheit is ',fahrenheit)
#calling funtion temp
temp()

#define number until it should make even and odd list
n=10
#define even,odd and total lists empty first
even=[]
odd=[]
total=[]
#for loop will run from 1 to 10
for j in range(1,n+1):
# if statement to find if the number is even 
    if(j%2==0):
#if true then append j with even list
        even.append(j)
# if false then else statement will be run and j will be appended with odd list    
    else:
        odd.append(j)
#print even and odd lists
print("even list: ",even)
print("odd list: ",odd)
#appending both lists to total
total.append(even)
total.append(odd)
#printing combined list
total
