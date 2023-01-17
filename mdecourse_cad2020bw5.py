# read http://mde.tw/cad2020/downloads/w5/cada_github_account.txt into dictionary
import urllib.request  # the lib that handles the url stuff
target_url = "http://mde.tw/cad2020/downloads/w5/cadb_github_account.txt"
# cad2b as a list
cad2b = []
for line in urllib.request.urlopen(target_url):
    # use utf-8 to decode and drop \n for each line
    cad2bTemp = line.decode('utf-8').rstrip()
    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is
    # data is seperated by \t, i.e. the tab key
    cad2b.append(cad2bTemp.split('\t'))
#print(cad2b)
# drop the first element of cp1b and convert into dictionary
cad2bAccount = dict(cad2b[1:])
# check into the cad2aAccount dict for "40823112"
print(cad2bAccount["40823231"])
import urllib.request
w3bGroup = [[40723224, 40823204, 40823212, 40823215, 40823219, 40823225, 40823226, 40823234, 40823250, 40823251, 40832244], [40823211, 40823213, 40823214, 40823222, 40823229, 40823230, 40823231, 40823236, 40823242, 40823245], [40723135, 40723215, 40823201, 40823203, 40823205, 40823206, 40823210, 40823217, 40823227, 40823248], [40823207, 40823208, 40823216, 40823218, 40823220, 40823224, 40823228, 40823238, 40823244, 40823246], [40823202, 40823221, 40823223, 40823232, 40823233, 40823235, 40823237, 40823239, 40823241, 40823247]]
# check if the student is still registered in the course
# 2b student list from registar server
target_url = "https://nfulist.herokuapp.com?semester=1091&courseno=0801"
# 2b registered student list
cad2bReg = []
for line in urllib.request.urlopen(target_url):
    cad2bReg.append(line.decode('utf-8').rstrip())
#print(cad2bReg)
# generate https://github.com/ + account + cad2020 and https:// + account + .github.io/cad2020
# read in w3bGroup
for gpNum in range(len(w3bGroup)):
    # w3bGroup[gpNum] is the member list of group number (gpNum + 1) 
    if gpNum != 0:
        print("<br />"*2)
        print("==============================")
        print("<br />"*2)
    print("group " + str(gpNum + 1) + ":" + "<br />"*2)
    for i in range(len(w3bGroup[gpNum])):
        memberNum = w3bGroup[gpNum][i]
        # from number to check account
        memberAccount = cad2bAccount[str(memberNum)]
        #print(memberAccount)
        print("Repository: <a href='https://github.com/" + str(memberAccount) + "/cad2020'>" + str(memberNum) + "</a> | ", end="")
        print("Site: <a href='https://" + str(memberAccount) + ".github.io/cad2020'>" + str(memberNum) + "</a><br />")
        