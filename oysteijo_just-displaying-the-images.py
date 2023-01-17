from IPython.display import Image, display



from subprocess import check_output

print(check_output(["ls", "../input"]).decode("utf8"))
print(check_output(["ls", "../input/Train/"]).decode("utf8"))
print(check_output(["ls", "../input/TrainDotted/"]).decode("utf8"))
from IPython.display import Image, display

for x in range(11):

    display(Image("../input/Train/%s.jpg" % x))
from IPython.display import Image, display

for x in range(11):

    display(Image("../input/TrainDotted/%s.jpg" % x))