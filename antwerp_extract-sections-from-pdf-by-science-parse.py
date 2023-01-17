!wget https://github.com/allenai/science-parse/releases/download/v2.0.3/science-parse-cli-assembly-2.0.3.jar
!mkdir json

!java -Xmx6g -jar science-parse-cli-assembly-2.0.3.jar ../input/cvpr-2019-papers/CVPR2019/papers -o ./json
from glob import glob

jsons = glob("./json/*")

print("Num of output jsons:", len(jsons))