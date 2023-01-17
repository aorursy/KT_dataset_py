import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt # draw graph
import seaborn as sns # draw graph
from scipy.stats import norm
import warnings
warnings.filterwarnings(action='ignore')
import os
# Kiểm tra file trong thư mục input
print(os.listdir("../input"))
# Đọc dữ liệu sang DataFrame
data_file = "../input/Iris.csv"
iris = pd.read_csv(data_file)
iris.info()
# Đổi tên column
iris.columns=['Id', 'SL', 'SW', 'PL', 'PW', 'Species']
iris.head()
iris.info()
print("\nAll unique values of Species column:")
print(iris.Species.unique())
# Missing values
print('Missing values:', iris.isnull().sum(), sep = '\n')
print('\nMin values:',iris.describe().loc['min'], sep = '\n')
print('\nAll Species values:', iris.Species.unique())
# Tách dữ liệu thành 2 phần thuộc tính và nhãn
iris_X = iris.drop(columns = ['Id', 'Species'])
iris_Lb = iris[['Species']]
iris_X.head()
# Chuyển nhãn từ tên loại hoa sang kiểu số để dễ xử lý
label_values = iris_Lb.Species.unique().tolist()
Y_values = [0, 1, 2]
label_map = pd.Series(Y_values, index=label_values)
# Map sang kiểu số lưu vào dataframe iris_Y
iris_Y = iris_Lb.applymap(lambda x: label_map[x])
pd.concat([iris_Y[:3], iris_Y[50:53], iris_Y[-3:]])
iris_X.describe()
iris_X0 = iris_X # Lưu dữ liệu gốc vào iris_X0 trước khi normalization
# Data normalization min-max
def norm_min_max(data):
    min_col = data.describe().loc['min']
    max_col = data.describe().loc['max']  
    return (data - min_col) / (max_col - min_col)
def norm_z_score(data):
    mean_col = data.describe().loc['mean']
    std_col = data.describe().loc['std']  
    return (data - mean_col) / std_col
# normalization iris_X
iris_X = norm_min_max(iris_X0)
#iris_X = norm_z_score(iris_X0)
iris_X.describe()
# Đồ thị mật độ phân bố các giá trị theo thuộc tính
plt.figure()
sns.distplot(iris_X['SL'], hist=False, kde_kws={"color": "g", "lw": 2, "label": "SL"},)
sns.distplot(iris_X['SW'], hist=False, kde_kws={"color": "r", "lw": 2, "label": "SW"},)
sns.distplot(iris_X['PL'], hist=False, kde_kws={"color": "b", "lw": 2, "label": "PL"},)
sns.distplot(iris_X['PW'], hist=False, kde_kws={"color": "k", "lw": 2, "label": "PW"}, axlabel='SL - SW - PL - PW')
plt.title('Hàm mật độ phân bố đặc điểm hoa Iris')
plt.show()
# Hàm vẽ bản đồ phân bố các loại hoa theo thuộc tính
def iris_display(X, Y, col1 = 0, col2 = 1, xlabel = '', ylabel = '', title = ''):
    if type(X) is pd.core.frame.DataFrame: X = X.values
    if type(Y) is pd.core.frame.DataFrame: Y = Y.values
    lb_list = list(set(Y[:, 0])) # tên nhóm các loại hoa iris
    colors = ['b^', 'go', 'rs']
    for i, lb in enumerate(lb_list):
        x = X[Y[:,0] == lb, :]
        plt.plot(x[:, col1], x[:, col2], colors[i], markersize = 4, alpha = .8, label = lb_list[i])
    if (type(Y[0,0]) is str): plt.legend()  # show name of cluster
    plt.axis('equal')
    plt.plot()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.show()
iris_display(iris_X.values, iris_Lb, 0, 1, 'SL', 'SW', 'Biểu đồ phân bố theo SL-SW')
iris_display(iris_X.values, iris_Lb, 2, 3, 'PL', 'PW', 'Biểu đồ phân bố theo PL-PW')
# Đồ thị plotwise từng cặp đặc điểm
import seaborn as sns; sns.set(style="ticks", color_codes=True)
g = sns.pairplot(pd.concat([iris_X, iris_Lb], axis=1, sort=False), hue="Species", markers=["o", "s", "^"])
plt.show()
# Chuyển sang tọa độ PCA giảm xuống 2 chiều nhiều dữ liệu nhất
from sklearn.decomposition import PCA # import PCA function
pca = PCA(n_components=2)
# Chuyển sang tọa độ mới (2 chiều)
iris_X_pca = pca.fit(iris_X).transform(iris_X)
# 2 vector cơ sở mới
pca.components_
sum(pca.explained_variance_ratio_)
iris_display(iris_X_pca, iris_Lb, 0, 1, 'x1', 'x2', 'Biểu đồ phân bố theo PCA')
# Chuyển sang LDA giảm xuống 2 chiều phân loại tốt
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
# Chuyển sang tọa độ mới (2 chiều)
iris_X_lda = lda.fit(iris_X, iris_Y).transform(iris_X)
lda.scalings_ # vector cơ sở tọa độ mới
iris_display(iris_X_lda, iris_Lb, 0, 1, 'x1', 'x2', 'Biểu đồ phân bố theo LDA')
# Phần 2: Phân nhóm theo thuật toán K-Means Clustering
from scipy.spatial.distance import cdist # import hàm tính khoảng cách
def kmeans_init_centers(X, k):     # tạo K điểm trung tâm ngẫu nhiên từ những mẫu ban đầu
    return X[np.random.choice(X.shape[0], k, replace=False)]

def kmeans_assign_labels(X, centers): # gán nhán mới cho các điểm theo center gần nhất
    D = cdist(X, centers)   # tính khoảng cách của X với các điểm trung tâm
    return np.argmin(D, axis = 1) # trả về index điểm trung tâm gần nhất

def kmeans_update_centers(X, labels, K): # cập nhật các điểm trung tâm 
    centers = np.zeros((K, X.shape[1]))
    for k in range(K):
        Xk = X[labels == k, :]  # nhóm các điểm thuộc cluster thứ k
        centers[k,:] = np.mean(Xk, axis = 0) # cập nhật điểm trung tâm
    return centers

def has_converged(centers, new_centers): # kiểm tra điều kiện dừng của thuật toán
    # kiểm tra tập hợp điểm trung tâm mới có trùng với tập hợp điểm cũ không
    return (set([tuple(a) for a in centers]) == 
        set([tuple(a) for a in new_centers]))
# Viết hàm kmeans
def kmeans(X, K): # X: dataframe, K:số cluster cần chia
    if type(X) is pd.core.frame.DataFrame:
        X = X.values
    centers = [kmeans_init_centers(X, K)]
    labels = []
    it = 0 # lưu số bước lặp đến khi dừng thuật toán
    while True:
        labels.append(kmeans_assign_labels(X, centers[-1]))
        new_centers = kmeans_update_centers(X, labels[-1], K)
        if has_converged(centers[-1], new_centers):
            break
        centers.append(new_centers)
        it += 1
    return (centers, labels, it) # trả về tọa độ điểm trung tâm, nhãn các cluster, số bước lặp
# Hàm đảo các giá trị cụm
def swap_label(labels, vl1, vl2):
    for i, lb in enumerate(labels):
        if lb == vl1: labels[i] = vl2
        elif lb == vl2: labels[i] = vl1

# Tính số nhãn dự đoán đúng
def n_true_predict(predict_labels):
    truelbs = iris_Y.values[:,0]
    if hasattr(predict_labels, 'values'): predict_labels = predict_labels.values[:,0]
    uv_true = np.unique(truelbs)  # unique value in true label
    uv_lb = np.unique(predict_labels) # unique value in predict label
    tb = pd.DataFrame(0, columns=uv_true, index=uv_true) # tạo bảng khớp các nhãn
    for i, ut in enumerate(uv_true):
        for j, ul in enumerate(uv_lb):
            tb.iloc[i, j] = (np.logical_and(truelbs == ut, predict_labels == ul)).sum()
    # khớp các nhãn để số phần tử đúng cao nhất
    i = 0
    while i < len(uv_true) - 1:
        for j in range(i + 1, (len(uv_lb))):
            if  (tb.iloc[i][j] + tb.iloc[j][i] > tb.iloc[i][i] + tb.iloc[j][j]):
                # swap column i, j
                tb.iloc[:, i] += tb.iloc[:, j]
                tb.iloc[:, j] = tb.iloc[:, i] - tb.iloc[:, j]
                tb.iloc[:, i] = tb.iloc[:, i] - tb.iloc[:, j]
                swap_label(predict_labels, i, j)
                i = -1; break;  # set i = -1 duyệt lại từ đầu
        i += 1
    n_tp = 0
    for i in range(len(uv_true)): n_tp += tb.iloc[i, i] # số dự đoán đúng
    return n_tp # trả về số nhãn phân cụm đúng
# khai báo 3 tham số dataframe sẽ khảo sát
dataFrames = [iris_X_pca, iris_X, iris_X.iloc[:, 2:4]]
dataFrames_tiles = ['PCA', 'FULL', 'PL-PW']
# Truyền các bộ tham số để phân cụm
K = 3 # chia làm 3 cluster
iris_results_kmeans = {}
plt.figure(figsize=(7, 5))
def Kmeans_Run():
    for i, df in enumerate(dataFrames):
        rsname = dataFrames_tiles[i]
        title = 'K-means Clustering (' + rsname + ')'
        iris_results_kmeans[rsname] = kmeans(df, K)
        (centers, labels, it) = iris_results_kmeans[rsname]  # Lấy kết quả (centers, labels, it) cuối cùng khi hội tụ
        predict_label = labels[-1]
        n_tp = n_true_predict(labels[-1])
        rstable = pd.crosstab(iris_Y.values[:,0], predict_label, rownames = ['label'], colnames = ['predict'])

        # Hiển thị centers màu vàng
        plt.scatter(centers[-1][:, -2], centers[-1][:,-1], s = 100, c = 'yellow', label = 'Centroids')
        # Hiển thị độ chính xác việc phân cụm
        acc_text = 'Accuracy: ' +  str(round(n_tp / 150 * 100, 2)) + '%\n' + str(rstable) 
        plt.figtext(0.92, 0.15, acc_text, fontname="DejaVu Sans Mono")
        # Hiển thị mật độ phân bố các cụm
        iris_display(df, pd.DataFrame(data=predict_label), -2, -1, '', '', title)
Kmeans_Run()
# Phần 3: Phân nhóm theo pp Hierarchical clustering
from collections import Counter
from scipy.cluster.hierarchy import dendrogram, linkage, cut_tree
def hierarchical_clustering(df, distance_method): # Hàm phân cụm
    linkage_matrix = linkage(df, distance_method) # tạo ra bảng linkage matrix
    predict_label = cut_tree(linkage_matrix, n_clusters=3) # nhãn dự đoán khi cut ra 3 cụm
    # Đổi nhãn 2 cụm 1, 2 nếu tỉ lệ nhận diện đúng thấp - nguyên nhân do lúc đánh số cụm 1, 2 sai.
    if (predict_label[50:100] == iris_Y[50:100]).sum()[0] < (3 - predict_label[50:100] == iris_Y[50:100]).sum()[0]:
        predict_label[predict_label > 0] = 3 - predict_label[predict_label > 0]
    # Bảng lưu số lượng các nhãn dự đoán đúng vơi từng cụm
    rstable = pd.crosstab(iris_Y.values[:,0], predict_label[:,0], rownames = ['label'], colnames = ['predict'])
    n_true_predict = (predict_label == iris_Y).sum()[0] # số lượng phân cụm đúng nhãn
    return (n_true_predict,rstable, predict_label, linkage_matrix)
# Hàm hiển thị dendrogram, tỉ lệ Accuracy
def show_dendrogram(linkage_matrix, title, text = ''):
    plt.figure(figsize=(7, 5))
    dendrogram(
        linkage_matrix,
        truncate_mode='lastp',  # chỉnh chế độ hiển thị merge các mẫu
        p=10,  # chỉnh ngưỡng dendrogram gộp các mẫu
        color_threshold = linkage_matrix[-2][2], # tạo ngưỡng hiển thị 3 màu khác nhau ở 3 cây con cuối cùng
    )
    #plt.title('Hierarchical Clustering Dendrogram')
    plt.figtext(.65, .65, text, fontname="DejaVu Sans Mono") # Hiển thị độ chính xác việc phân cụm
    plt.title(title)
    plt.xlabel('sample index')
    plt.ylabel('distance')
    plt.tight_layout()
    plt.show()
# Truyền các bộ tham số để phân cụm
distance_methods = ['single', 'average', 'complete'] # ứng với hàm tính khoảng cách theo gần nhất, trung bình, xa nhất
iris_results_hierarchical = {}
def Hierarchical_Run():
    for i, df in enumerate(dataFrames):
        for mt in distance_methods:
            title = 'Hierarchical Clustering Dendrogram (' + dataFrames_tiles[i] + ', ' + mt + ')'
            rsname = (dataFrames_tiles[i] + ',' + mt).replace(" ", "")
            # cho phép sau này gọi lại bằng tên frame + distance_method. VD: iris_results_hierarchical['PCA,average']
            iris_results_hierarchical[rsname] = hierarchical_clustering(df, mt)
            # Độ phân cụm chính xác
            acc_text = 'Accuracy: ' + str(round(iris_results_hierarchical[rsname][0] / 150 * 100, 2)) + '%\n' + str(iris_results_hierarchical[rsname][1])
            show_dendrogram(iris_results_hierarchical[rsname][3], title, acc_text)
Hierarchical_Run()
iris_display(iris_X.iloc[:, 2:], iris_results_hierarchical['PL-PW,average'][2], 0, 1, '(PL-PW, average)', '', 'Biểu đồ Hierarchical Clustering theo PL-PW')
iris_display(iris_X_pca, iris_results_hierarchical['PCA,average'][2], 0, 1, '(PCA, average)', '', 'Biểu đồ Hierarchical Clustering theo PCA')