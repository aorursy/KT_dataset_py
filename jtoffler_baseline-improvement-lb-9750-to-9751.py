import pandas as pd



def dateFixer(x):

    return x.replace('two thousand and sixteen', 'две тысячи шестнадцатого').replace('two thousand and seventeen', 'две тысячи семнадцатого').replace('eleven', 'одиннадцатого').replace('twelve', 'двенадцатого').replace('thirteen', 'тринадцатого').replace('fourteen', 'четырнадцатого').replace('fifteen', 'пятнадцатого').replace('sixteen', 'шестнадцатого').replace('seventeen', 'семнадцатого').replace('eighteen', 'восемнадцатое').replace('nineteen', 'девятнадцатого').replace('twenty-one', 'двадцать первого').replace('twenty-two', 'двадцать второго').replace('twenty-three', 'двадцать третьего').replace('twenty-four', 'двадцать четвертого').replace('twenty-five', 'двадцать пятого').replace('twenty-six', 'двадцать шестого').replace('twenty-seven', 'двадцать седьмого').replace('twenty-eight', 'двадцать восьмого').replace('twenty-nine', 'двадцать девятого').replace('twenty', 'двадцатого').replace('thirty-one', 'тридцать первого').replace('thirty', 'тридцатого').replace('two', 'второго').replace('three', 'третьего').replace('four', 'четвертого').replace('five', 'пятого').replace('six', 'шестого').replace('seven', 'седьмого').replace('eight', 'восьмого').replace('nine', 'девятого').replace('one', 'первого').replace('ten', 'десятого')



results = pd.read_csv("../input/baseline-results/baseline_ext_ru.csv")

newColumn = results['after'].apply(dateFixer)

results['after'] = newColumn

results.to_csv("submission.csv", index = False)