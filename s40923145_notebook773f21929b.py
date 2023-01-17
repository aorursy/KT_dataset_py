for i in range(5):
    print ("i = "+str(i))
import urllib.request  # the lib that handles the url stuff
target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0762"
cp1a = []
for line in urllib.request.urlopen(target_url):
    cp1a.append(line.decode('utf-8').rstrip())
    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is
print(cp1a)
'''
<a href="https://github.com/40923145/cp2020">40923145</a>

'''