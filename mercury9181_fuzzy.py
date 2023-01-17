print("Temperature range is 0 to 40")
print("Humidity range is 0 to 100")
t= int(input("Enter temperature: "))
h= int(input("Enter humidity: "))
fuzzified_temp= str()

if t in range(36,41):
    fuzzified_temp="hottest"
elif t in range(30,36):
    fuzzified_temp="hot"
elif t in range(21,30):
    fuzzified_temp="warm"
elif t in range(10,21):
    fuzzified_temp="cold"
elif t in range(0,10):
    fuzzified_temp="coldest"
print(t,fuzzified_temp)
    
fuzzified_hum= str()

if h in range(60,101):
    fuzzified_hum="high"
elif h in range(30,60):
    fuzzified_hum="optimal"
elif h in range(0,30):
    fuzzified_hum="low"

print(h,fuzzified_hum)
knob_1= "Tuning the AC to very cold mode...."
knob_2= "Tuning the AC to cold mode...."
knob_3= "Tuning the AC to no change mode...."
knob_4= "Tuning the AC to warm mode...."
knob_5= "Tuning the AC to very warm mode...."
decision= str()
if fuzzified_temp=="hottest" and fuzzified_hum=="high":
    decision = knob_1
elif fuzzified_temp=="hottest" and fuzzified_hum=="optimal":
    decision = knob_1
elif fuzzified_temp=="hottest" and fuzzified_hum=="low":
    decision = knob_1
elif fuzzified_temp=="hot" and fuzzified_hum=="high":
    decision = knob_2
elif fuzzified_temp=="hot" and fuzzified_hum=="optimal":
    decision = knob_2
elif fuzzified_temp=="hot" and fuzzified_hum=="low":
    decision = knob_2
elif fuzzified_temp=="warm" and fuzzified_hum=="high":
    decision = knob_2
elif fuzzified_temp=="warm" and fuzzified_hum=="optimal":
    decision = knob_3
elif fuzzified_temp=="warm" and fuzzified_hum=="low":
    decision = knob_3
elif fuzzified_temp=="cold" and fuzzified_hum=="high":
    decision = knob_4
elif fuzzified_temp=="cold" and fuzzified_hum=="optimal":
    decision = knob_3
elif fuzzified_temp=="cold" and fuzzified_hum=="low":
    decision = knob_4
elif fuzzified_temp=="coldest" and fuzzified_hum=="high":
    decision = knob_5
elif fuzzified_temp=="coldest" and fuzzified_hum=="optimal":
    decision = knob_4
elif fuzzified_temp=="coldest" and fuzzified_hum=="low":
    decision = knob_5
print("Room temperature is " + str(t))
print("Room humidity is " + str(h))
print(decision)