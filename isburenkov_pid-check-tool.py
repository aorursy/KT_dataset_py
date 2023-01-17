import psutil



for proc in psutil.process_iter():

        print(proc)
PID = 1



if PID in psutil.pids():

    print("Process with pid =", PID, "is currentrly running")

else:

    print("Process was not found")
PID = 2



if PID in psutil.pids():

    print("Process with pid =", PID, "is currentrly running")

else:

    print("Process was not found")
PID = int(input("Enter your PID: "))



if PID in psutil.pids():

    print("Process with pid =", PID, "is currentrly running")

else:

    print("Process was not found")
