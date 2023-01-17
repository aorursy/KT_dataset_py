import numpy as np
x=np.sqrt(2)**2-2
print("Result=",x)
y=round(np.sqrt(2)**2-2,2)
print("Result (Rounded)=",y)
%time sum(range(100))
%timeit L = [n ** 2 for n in range(1000)]
def sum_of_lists(N):
		total = 0
		for i in range(100):
			L = [j ^ (j >> i) for j in range(N)]
			total += sum(L)
		return total
	
%prun sum_of_lists(50)
%prun sum_of_lists(10000)
%load_ext line_profiler
%lprun -f sum_of_lists sum_of_lists(50)
%lprun -f sum_of_lists sum_of_lists(500000)
%load_ext memory_profiler
%memit sum_of_lists(1000000)
%memit sum_of_lists(100)
