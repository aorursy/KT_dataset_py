def sarea(s):
    print("Area of the  given square with  side ",s,"is",s**2)
def rarea(ln,bd):
    print("Area of the given rectangle with length",ln,"and breadth",bd,"is",ln*bd)
def tarea(a,b,c):
    import math
    try:
        p=(a+b+c)/2
        ar=math.sqrt((p*(p-a)*(p-b)*(p-c)))
        print("Area of triangle with sides ",a,b,c,"are",ar)
    except ValueError:
        print("Unable to compute area of triangle as the Avg. of the sides is lesser than any of the entered sides.")
        print("Please enter proportinate length of the sides")
def tarea1(base,height):
    ar2=(1/2)*base*height
    print("Area of triangle with given base",base,"and height",height,"is",ar2)
def carea(r):
    print("Area of circle with given radius ",r,"is",(3.14*r*r))
print("Hello!!! Welcome to Shape Area Calculator.\nBelow are the option numbers for the respective shapes.")        
print("\n1.Square\t2.Rectangle\n3.Triangle\t4.Circle\n")    
while(1):
        op=(input("Enter an option number for the shapes:"))
        print()
        if(int(op)==1):
            print("You have selected the shape Rectangle.\n")
            print("Enter length of side\n")
            s=float(input())
            sarea(s)
        elif(int(op)==2):
            print("Enter the length of rectangle:")
            ln=float(input())
            print("Enter the breadth of rectangle:")
            bd=float(input())
            rarea(ln,bd)
        elif(int(op)==3):
            print("\n1.Enter side\t2.Enter base and height\n")
            op1=float(input())
            if(op1==1):
                print("Enter the sides of triangle\n")
                a=float(input("Enter length of side 1:"))
                b=float(input("Enter length of side 2:"))
                c=float(input("Enter length of side 3:"))
                tarea(a,b,c)
            elif(op1==2):
                print("Enter base and height\n")
                base=float(input())
                height=float(input())
                tarea1(base,height)
        elif(int(op)==4):
            print("Enter radius of circle\n")
            r=float(input())
            carea(r)
        else:
            print("Invalid input, please enter a number in range [1 to 4].\nFor Example:Just enter 1 if you want to calculate area of Square:")
            break
        print()
        print()
        print("Do you want to continue:y/n")
        te=input()
        if(te=='y'):
            pass
        elif(te=='n'):
            print()
            print("Thank you for using the Area Shape Calculator")
            break
        else:
            print("I do not understand your command:",te)
            print("please enter y for Yes, n for No")
            print("Run the tool again!!!")
            break

