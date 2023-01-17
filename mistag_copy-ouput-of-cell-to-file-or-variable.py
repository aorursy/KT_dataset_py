%%capture cap_out --no-stderr



for i in range (10):

    print("Capture me line {}".format(i))

txt = cap_out.stdout

print(txt)
import sys



old_stdout = sys.stdout # keep reference to existing stdout

sys.stdout = open('logfile.txt', 'w')



for i in range (10):

    print("Log me line {}".format(i))



sys.stdout = old_stdout # restore stdout
!ls
!cat logfile.txt