

ones = {

1:"one",2:"two",3:"three",4:"four",

5:"five",6:"six",7:"seven",8:"eight",

9:"nine",0:"zero",10:"ten"

}



teens = {

11:"eleven",12:"twelve",13:"thirteen",

14:"fourteen",15:"fifteen"

}



tens = {

2:"twenty",3:"thirty",4:"forty",

5:"fifty",6:"sixty",7:"seventy",

8:"eighty",9:"ninety"

}



powoftens = {

3:"hundred",4:"thousand",6:"hundred",7:"million",

8:"million", 9:"million",10:"billion"#,13:"trillion",11:"googol",

 }



def num2word (num):

    if num > 999999999:

        return "Number more than 1 billion"

    # Ones

    if num < 11:

        return ones[num]

    # Teens

    if num < 20:

        word = ones[num%10] + "teen" if num > 15 else teens[num]

        return word

    # Tens

    if num > 19 and num < 100:

        word = tens[int(str(num)[0])]

        if str(num)[1] == "0":

            return word

        else:

            word = word + " " + ones[num%10]

            return word



    # First digit for thousands,hundred-thousands.

    if len(str(num)) in powoftens and len(str(num)) != 3:

        word = ones[int(str(num)[0])] + " " + powoftens[len(str(num))]

    else:

        word = ""



    # Hundred to Million

    if num < 1000000:

        # First and Second digit for ten thousands.

        if len(str(num)) == 5:

            word = num2word(int(str(num)[0:2])) + " thousand"

        # How many hundred-thousand(s).

        if len(str(num)) == 6:

            word = word + " " + num2word(int(str(num)[1:3])) + " " + powoftens[len(str(num))-2]

        # How many hundred(s)?

        thousand_pt = len(str(num)) - 3

        word = word + " " + ones[int(str(num)[thousand_pt])] + " " + powoftens[len(str(num))-thousand_pt]

        # Last 2 digits.

        last2 = num2word(int(str(num)[-2:]))

        if last2 != "zero":

            word = word + " and " + last2

        word = word.replace(" zero hundred","")

        return word.strip()



    left, right = '',''

    # Less than 1 million.

    if num < 100000000:

        left = num2word(int(str(num)[:-6])) + " " + powoftens[len(str(num))]

        right = num2word(int(str(num)[-6:]))

    # From 1 million to 1 billion.

    if num > 100000000 and num < 1000000000:

        left = num2word(int(str(num)[:3])) +  " " + powoftens[len(str(num))]

        right = num2word(int(str(num)[-6:]))

    if int(str(num)[-6:]) < 100:

        word = left + " and " + right

    else:

        word = left + " " + right

    word = word.replace(" zero hundred","").replace(" zero thousand"," thousand")

    return word

num2word(int(923159))
import pandas as pd
df = pd.read_csv('../input/en_test.csv')



previous = ""

for idx, row in df.iterrows():

    token = row['before']

    try:

        numbernated = num2word(int(token))

        print(prev, token, '\t', prev, numbernated)

    except:

        pass

    previous = token