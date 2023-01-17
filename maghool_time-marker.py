import time



a=0

b=1

r=0

r=int(input("Press '1' for new time, '2' for the next subject, '3' for reset, '9' for exit"))



while True:

        if r!=2 and r!=9:

            a=a+1

            millis = int(round(time.time() * 1000))

            print ("subject id: " + str(b) + "  marker id: " + str(a))

            print (millis)

            time.sleep(0.1)

            r=int(input())

                        

        if r==2:

            b=b+1

            a=0

            millis = int(round(time.time() * 1000))

            print ("subject id: " + str(b) + "  marker id: " + str(a))

            print (millis)

            time.sleep(0.1)  

            r=int((input()))

            

        if r==3:

            b=1

            a=0

            millis = int(round(time.time() * 1000))

            print ("subject id: " + str(b) + "  marker id: " + str(a))

            print (millis)

            time.sleep(0.1)  

            r=int((input()))

            

        if r==9:

            exit()

            

    
