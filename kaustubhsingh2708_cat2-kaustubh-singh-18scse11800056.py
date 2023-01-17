#CAT 2(Kaustubh Singh 18SCSE1180056)
from matplotlib import pyplot as plt
data=[36,25,38,46,55,68,72,55,36,38,67,45,22,48,91,46,52,61,58,55]
bins=[20,30,40,50,60,70,80,90,100]
plt.hist(data,bins,histtype="bar",rwidth=0.8,color="c")
plt.title("Histogram")
plt.legend()
plt.show()