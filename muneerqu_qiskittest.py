from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister, execute, Aer
from qiskit.tools.visualization import plot_histogram

qr = QuantumRegister(2)
cr = ClassicalRegister(2)

qc = QuantumCircuit(qr, cr)

qc.x(qr[0])
qc.h(qr[0])
qc.cx(qr[0],qr[1])
qc.measure(qr,cr)

backend = Aer.get_backend('qasm_simulator')
job = execute(qc, backend=backend)
result = job.result()

plot_histogram(result.get_counts())
