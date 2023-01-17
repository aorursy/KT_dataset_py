import multiprocessing



print("CPUs", multiprocessing.cpu_count())
def get_size(bytes, suffix="B"):

    factor = 1024

    for unit in ["", "K", "M", "G", "T", "P"]:

        if bytes < factor:

            return f"{bytes:.2f}{unit}{suffix}"

        bytes /= factor



import psutil



svmem = psutil.virtual_memory()

print(f"Total: {get_size(svmem.total)}")

print(f"Available: {get_size(svmem.available)}")

print(f"Used: {get_size(svmem.used)}")

print(f"Percentage: {svmem.percent}%")