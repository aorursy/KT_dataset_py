for i in range(5):
    print("i="+ str(i))
import urllib.request  # the lib that handles the url stuff
target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0762"
cp1a = []
for line in urllib.request.urlopen(target_url):
    cp1a.append(line.decode('utf-8'))
    #print(line.decode('utf-8'), end = "") #utf-8 or iso8859-1 or whatever the page encoding scheme is
print(cp1a)
a = [1, 2, 3, 4]
a_len = len(a)
for i in range(a_len):
    print(a[i])
print(a[1:])
import urllib.request
target_url = "https://nfulist.herokuapp.com/?semester=1091&courseno=0762"
for line in urllib.request.urlopen(target_url):
    print(line.decode('utf-8'))