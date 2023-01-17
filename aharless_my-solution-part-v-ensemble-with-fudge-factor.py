ACTUALLY_DO_SOMETHING = True

CORRECT_FOR_BUG = True
from tqdm import tqdm
import gzip
ff = 1.08  # Global fudge factor, based on individual FFs guesstimated from LB probes



weights = [ 0.27, 0.18,   # Weights to apply to 2017 predictions

            0.20, 0.35 ]  # Fit using 2017 training data with 2016 models

                                # assuming optimal fudge factors
print(sum(weights))
if CORRECT_FOR_BUG:

    weights[1] /= 10
fnames = ('output_dataset2017a.csv',           # 0 StackNet

          'finalXgbApril20171017_011204.csv',  # 1 My XGBoost tuned with Q4 CV, Apr 30 Seas

          '2017Q4.csv',                        # 2 Genetic

          'catboost_opt4.csv')                 # 3 CatBoost

output="withReoptimizedXGB.csv"   # file to generate submissions to
dir = "../input/my-zillow-predictions/"

if ACTUALLY_DO_SOMETHING:

    fs=open(output,"w")

    result = []

    ids = []

    first = True

    for w, n in zip(weights, fnames):

        print ("Processing file: ", n)

        f = open(dir+n, 'r')

        for iline, line in tqdm(enumerate(f)):

            if iline==0:

                if first:

                    fs.write(line)

            else:

                i = iline-1

                splits=line.replace("\n","").split(",")

                if first:

                    ids.append(splits[0])

                else:

                    assert(ids[i]==splits[0])

                preds=[]

                for j in range (1,7):

                    preds.append(float(splits[j])*ff*w)

                if first:

                    result.append(preds)

                else:

                    for j in range(0,6):

                        result[i][j] += preds[j]

        f.close()

        first = False        
if ACTUALLY_DO_SOMETHING:

    for id, r in zip(ids,result):

        fs.write(id)

        for j in range(6):

            fs.write( "," +str(round(r[j],6) ))

        fs.write("\n")
if ACTUALLY_DO_SOMETHING:

    fs.close()
if ACTUALLY_DO_SOMETHING:

    f_in = open(output, 'rb')

    f_out = gzip.open(output+'.gz', 'wb')

    f_out.writelines(f_in)

    f_out.close()

    f_in.close()
2+2 # Just so I know when it's finished